"""Generate numeric, adjacent-frame GelSight flow without training-code imports.

Input in each episode directory:
    tactile_rgb.npz: finger1, finger2, finger3, each uint8 [T,H,W,3] in RGB order
    signals.csv: one row per RGB frame, with a strictly increasing time in seconds

PNG RGB sequences are also supported: rgb_png/finger1/frame_0000.png, etc.
The CLI recognizes per-episode directories and run-level directories grouped by
format (rgb_png, signals_csv, flow_npz), without pairing different episodes.

Output tactile_flow.npz contains float32 [T-1,H,W,2] for each finger. Channel 0
is dx (right positive); channel 1 is dy (down positive). Values are displacement
in pixels per frame pair, NOT pixels/second. The aligned dt array permits callers
to calculate velocity explicitly. Flow is estimated from the saved RGB images
at their original resolution; it is not a simulator ground-truth deformation.

Example:
    python tactile_optical_flow.py --root path/to/dataset
    python tactile_optical_flow.py --root path/to/dataset --overwrite

An existing result is reused only if source content, algorithm, OpenCV version,
and frame alignment match. A stale result requires explicit --overwrite.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any
import zipfile

import numpy as np


FINGERS = ("finger1", "finger2", "finger3")
SCHEMA_VERSION = 1
FLOW_UNITS = "pixels/frame_pair"
FARNEBACK_PARAMETERS = {
    "pyr_scale": 0.5,
    "levels": 3,
    "winsize": 21,
    "iterations": 5,
    "poly_n": 7,
    "poly_sigma": 1.5,
    "flags": 0,
}


def _require_cv2():
    try:
        import cv2
    except ImportError:
        # Reuse the same existing dependencies as the collector without loading
        # Taxim, MuJoCo, or any training code for standalone flow generation.
        root = Path(__file__).resolve().parent
        for candidate in (root / "third_party" / "_deps", root / "_deps",
                          Path(r"E:\project\p1\rl\新建文件夹\output\tactile_collection\_deps")):
            if candidate.is_dir() and str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "GelSight optical-flow generation requires OpenCV (cv2). "
                "Use a Python environment with opencv-python or opencv-python-headless; "
                "no optical-flow file was generated."
            ) from exc
    if not hasattr(cv2, "calcOpticalFlowFarneback"):
        raise RuntimeError("The installed cv2 has no calcOpticalFlowFarneback implementation.")
    return cv2


def _fingerprint(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return {"file": path.name, "sha256": digest.hexdigest(), "bytes": size}


def _png_frame_paths(path: Path) -> dict[str, list[Path]]:
    """Require one complete, synchronized sequence for each finger."""
    sequences = {}
    for finger in FINGERS:
        directory = path / finger
        if not directory.is_dir():
            raise ValueError(f"{path}: missing RGB sequence directory {finger}.")
        numbered = {}
        for image_path in directory.iterdir():
            if not image_path.is_file() or image_path.suffix.lower() != ".png":
                continue
            match = re.fullmatch(r"frame_([0-9]{4,})\.png", image_path.name)
            if not match:
                raise ValueError(f"{image_path}: expected PNG filename frame_0000.png.")
            frame = int(match.group(1))
            if frame in numbered:
                raise ValueError(f"{directory}: duplicate PNG frame index {frame}.")
            numbered[frame] = image_path
        indices = sorted(numbered)
        if len(indices) < 2 or indices != list(range(len(indices))):
            raise ValueError(
                f"{directory}: PNG frame indices must start at 0, be contiguous, "
                "and contain at least 2 frames."
            )
        sequences[finger] = [numbered[index] for index in indices]
    expected = len(sequences[FINGERS[0]])
    if any(len(paths) != expected for paths in sequences.values()):
        raise ValueError(f"{path}: three fingers must have synchronized PNG frame counts.")
    return sequences


def _rgb_fingerprint(path: Path) -> dict[str, Any]:
    if not path.is_dir():
        # Preserve the legacy source schema so existing valid NPZ caches work.
        return _fingerprint(path)
    sequences = _png_frame_paths(path)
    return {
        "format": "png_rgb",
        "directory": path.name,
        "files": [
            {**_fingerprint(image_path), "file": image_path.relative_to(path).as_posix()}
            for finger in FINGERS for image_path in sequences[finger]
        ],
    }


def _load_rgb(path: Path, cv2=None) -> dict[str, np.ndarray]:
    images = {}
    if path.is_dir():
        cv2 = cv2 or _require_cv2()
        for finger, paths in _png_frame_paths(path).items():
            frames = []
            for image_path in paths:
                # imread does not consistently support Unicode Windows paths.
                try:
                    bgr = cv2.imdecode(np.frombuffer(image_path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
                except cv2.error as exc:
                    raise ValueError(f"{image_path}: cannot decode an RGB PNG image.") from exc
                if bgr is None or bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[-1] != 3:
                    raise ValueError(f"{image_path}: cannot decode an RGB PNG image.")
                frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            if any(frame.shape != frames[0].shape for frame in frames):
                raise ValueError(f"{path}: {finger} PNG frames must have identical dimensions.")
            images[finger] = np.stack(frames)
    else:
        with np.load(path, allow_pickle=False) as archive:
            for finger in FINGERS:
                if finger not in archive.files:
                    raise ValueError(f"{path}: missing RGB sequence {finger}.")
                images[finger] = archive[finger]
    for finger in FINGERS:
        sequence = images[finger]
        if sequence.dtype != np.uint8 or sequence.ndim != 4 or sequence.shape[-1] != 3:
            raise ValueError(
                f"{path}: {finger} must be uint8 [T,H,W,3] RGB; "
                f"got {sequence.dtype} {sequence.shape}."
            )
        if sequence.shape[0] < 2 or any(length < 1 for length in sequence.shape[1:3]):
            raise ValueError(f"{path}: {finger} requires at least 2 nonempty RGB frames.")
    expected = images[FINGERS[0]].shape
    for finger, sequence in images.items():
        if sequence.shape != expected:
            raise ValueError(
                f"{path}: three fingers must have synchronized frame counts and dimensions; "
                f"{finger} has {sequence.shape}, expected {expected}."
            )
    return images


def _load_signals(path: Path, num_frames: int) -> tuple[np.ndarray, np.ndarray | None]:
    times = []
    contacts = []
    contact_columns = [f"contact_count_f{index}" for index in (1, 2, 3)]
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames or "time" not in reader.fieldnames:
            raise ValueError(f"{path}: signals.csv must contain a 'time' column in seconds.")
        has_frame = "frame" in reader.fieldnames
        has_contacts = all(name in reader.fieldnames for name in contact_columns)
        if any(name in reader.fieldnames for name in contact_columns) and not has_contacts:
            raise ValueError(f"{path}: contact counts require all three finger columns.")
        for index, row in enumerate(reader):
            try:
                times.append(float(row["time"]))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}: invalid time at data row {index + 1}.") from exc
            if has_frame:
                try:
                    frame = int(row["frame"])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{path}: invalid frame index at data row {index + 1}.") from exc
                if frame != index:
                    raise ValueError(
                        f"{path}: frame indices must start at 0 and be contiguous; "
                        f"row {index + 1} has frame {frame}, expected {index}."
                    )
            if has_contacts:
                try:
                    counts = [int(row[name]) for name in contact_columns]
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{path}: invalid contact count at data row {index + 1}.") from exc
                if any(count < 0 for count in counts):
                    raise ValueError(f"{path}: contact counts cannot be negative at data row {index + 1}.")
                contacts.append([count > 0 for count in counts])
    if len(times) != num_frames:
        raise ValueError(
            f"{path}: RGB/time synchronization mismatch: {num_frames} RGB frames "
            f"but {len(times)} signal rows."
        )
    time_array = np.asarray(times, dtype=np.float64)
    if not np.isfinite(time_array).all() or not np.all(np.diff(time_array) > 0):
        raise ValueError(f"{path}: frame times must be finite and strictly increasing.")
    return time_array, np.asarray(contacts, dtype=np.bool_) if has_contacts else None


def _alignment(times: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "frame_from": np.arange(len(times) - 1, dtype=np.int64),
        "frame_to": np.arange(1, len(times), dtype=np.int64),
        "time_start": times[:-1],
        "time_end": times[1:],
        "dt": np.diff(times),
    }


def _check_existing(
    path: Path, expected_metadata: dict[str, Any], alignment: dict[str, np.ndarray]
) -> None:
    """Validate the cached archive; never trust only its filename or modification time."""
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        if metadata != expected_metadata:
            raise ValueError("source content or optical-flow configuration has changed")
        if archive["units"].item() != FLOW_UNITS:
            raise ValueError("incorrect flow units")
        for name, expected in alignment.items():
            actual = archive[name]
            if actual.dtype != expected.dtype or not np.array_equal(actual, expected):
                raise ValueError(f"incorrect {name} alignment")
        expected_shape = tuple(expected_metadata["flow_shape_per_finger"])
        for finger in FINGERS:
            flow = archive[finger]
            if flow.dtype != np.float32 or flow.shape != expected_shape or not np.isfinite(flow).all():
                raise ValueError(f"invalid {finger} flow array")


def _atomic_savez(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary_path = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # A sibling temporary file keeps os.replace atomic on the same filesystem.
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{path.stem}.", suffix=".tmp", dir=path.parent, delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            np.savez_compressed(temporary, **arrays)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def save_episode_flow(
    episode_dir: Path, *, rgb_path: Path | None = None, signals_path: Path | None = None,
    output_path: Path | None = None, overwrite: bool = False
) -> dict[str, Any]:
    """Save synchronized flow for exactly one episode and return JSON-safe metadata.

    Invalid input, missing OpenCV, or a stale output raises an explicit exception.
    No source files or episode metadata.json are changed by this function.
    An identical, valid existing output is reused unless overwrite=True. Explicit
    source/output paths support grouped formats; rgb_path may be a PNG directory
    containing finger1, finger2, and finger3 or a legacy tactile_rgb.npz file.
    """
    cv2 = _require_cv2()
    episode_dir = Path(episode_dir)
    rgb_path = Path(rgb_path) if rgb_path is not None else episode_dir / "tactile_rgb.npz"
    signals_path = Path(signals_path) if signals_path is not None else episode_dir / "signals.csv"
    output_path = Path(output_path) if output_path is not None else episode_dir / "tactile_flow.npz"
    if output_path.resolve() in (rgb_path.resolve(), signals_path.resolve()):
        raise ValueError("Optical-flow output must not overwrite an RGB or signals source.")
    sources = {"rgb": _rgb_fingerprint(rgb_path), "signals": _fingerprint(signals_path)}
    images = _load_rgb(rgb_path, cv2)
    num_frames, height, width, _ = images[FINGERS[0]].shape
    times, contacts = _load_signals(signals_path, num_frames)
    alignment = _alignment(times)
    if contacts is not None:
        alignment.update({
            "contact_start": contacts[:-1],
            "contact_end": contacts[1:],
            "contact_pair": contacts[:-1] & contacts[1:],
        })
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "file": output_path.name,
        "algorithm": "opencv_farneback",
        "opencv_version": str(cv2.__version__),
        "parameters": dict(FARNEBACK_PARAMETERS),
        "source": sources,
        "fingers": list(FINGERS),
        "num_frames": num_frames,
        "frame_pairs": num_frames - 1,
        "rgb_shape_per_finger": [num_frames, height, width, 3],
        "flow_shape_per_finger": [num_frames - 1, height, width, 2],
        "dtype": "float32",
        "units": FLOW_UNITS,
        "time_units": "seconds",
        "channels": ["dx", "dy"],
        "coordinates": "x right positive, y down positive; source-frame pixel coordinates",
        "direction": "frame_from -> frame_to",
        "frame_index_base": 0,
        "preprocessing": "cv2.COLOR_RGB2GRAY, original saved image resolution, no resizing",
        "estimation": "estimated_from_synthetic_RGB; not physical ground truth or tracked marker displacement",
        "ground_truth": False,
        "pairing": "consecutive frames within this episode only",
        "contact_mask": {
            "available": contacts is not None,
            "column_order": list(FINGERS),
            "source_columns": ["contact_count_f1", "contact_count_f2", "contact_count_f3"],
            "definition": "contact_start/end: contact_count > 0; contact_pair: both endpoints in contact",
            "interpretation": "flow outside contact_pair is not a tactile contact-motion measurement",
        },
    }
    if output_path.exists() and not overwrite:
        try:
            _check_existing(output_path, metadata, alignment)
        except (ValueError, KeyError, OSError, EOFError, zipfile.BadZipFile) as exc:
            raise FileExistsError(
                f"{output_path}: existing flow is stale or invalid ({exc}); "
                "use overwrite=True or CLI --overwrite to replace it."
            ) from exc
        _assert_sources_unchanged(rgb_path, signals_path, sources)
        return {**metadata, "status": "reused"}

    arrays = dict(alignment)
    for finger in FINGERS:
        sequence = images[finger]
        flow_sequence = np.empty((num_frames - 1, height, width, 2), dtype=np.float32)
        previous_gray = cv2.cvtColor(sequence[0], cv2.COLOR_RGB2GRAY)
        for index in range(num_frames - 1):
            next_gray = cv2.cvtColor(sequence[index + 1], cv2.COLOR_RGB2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                previous_gray, next_gray, None, **FARNEBACK_PARAMETERS
            )
            if flow.shape != (height, width, 2) or not np.isfinite(flow).all():
                raise RuntimeError(f"Invalid flow returned for {finger}, pair {index} -> {index + 1}.")
            flow_sequence[index] = flow
            previous_gray = next_gray
        arrays[finger] = flow_sequence
    arrays["units"] = np.asarray(FLOW_UNITS)
    arrays["metadata"] = np.asarray(json.dumps(metadata, ensure_ascii=False, sort_keys=True))
    _assert_sources_unchanged(rgb_path, signals_path, sources)
    _atomic_savez(output_path, arrays)
    return {**metadata, "status": "saved"}


def _assert_sources_unchanged(rgb_path: Path, signals_path: Path, sources: dict[str, Any]) -> None:
    try:
        actual = {"rgb": _rgb_fingerprint(rgb_path), "signals": _fingerprint(signals_path)}
    except (ValueError, OSError) as exc:
        raise RuntimeError("RGB/signals source changed during flow generation; retry after collection finishes.") from exc
    if actual != sources:
        raise RuntimeError("RGB/signals source changed during flow generation; retry after collection finishes.")


def _discover_inputs(root: Path) -> list[dict[str, Path]]:
    """Find supported layouts and prefer PNG if an episode also has legacy RGB."""
    root = root.resolve()
    jobs = {
        path.parent: {"episode_dir": path.parent}
        for path in root.rglob("tactile_rgb.npz")
    }
    rgb_roots = set(root.rglob("rgb_png"))
    rgb_roots.update(parent for parent in (root, *root.parents) if parent.name == "rgb_png")
    for rgb_root in sorted(rgb_roots):
        if not rgb_root.is_dir():
            continue
        # Infer the structure from files, not the user-chosen run directory's
        # name (a grouped run may itself be named "episode_trial").
        by_episode = (rgb_root.parent / "signals_csv" / "signals.csv").is_file() or any(
            any((rgb_root / finger).glob("*.png")) for finger in FINGERS
        )
        candidates = [rgb_root] if by_episode else sorted(rgb_root.glob("*/episode_*"))
        for candidate in candidates:
            if not candidate.is_dir() or not (
                root == candidate or root in candidate.parents or candidate in root.parents
            ):
                continue
            if by_episode:
                episode = rgb_root.parent
                signals = episode / "signals_csv" / "signals.csv"
                output = episode / "flow_npz" / "tactile_flow.npz"
            else:
                relative_episode = candidate.relative_to(rgb_root)
                episode = rgb_root.parent / relative_episode
                signals = rgb_root.parent / "signals_csv" / relative_episode / "signals.csv"
                output = rgb_root.parent / "flow_npz" / relative_episode / "tactile_flow.npz"
            jobs[episode] = {
                "episode_dir": episode, "rgb_path": candidate,
                "signals_path": signals, "output_path": output,
            }
    return [jobs[key] for key in sorted(jobs)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True, help="Episode directory or dataset root to scan recursively.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing flow, including stale outputs.")
    args = parser.parse_args(argv)
    if not args.root.is_dir():
        parser.error(f"root is not a directory: {args.root}")
    try:
        _require_cv2()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    episodes = _discover_inputs(args.root)
    if not episodes:
        print(f"No supported RGB episodes found under {args.root}", file=sys.stderr)
        return 1
    counts = {"saved": 0, "reused": 0, "failed": 0}
    for paths in episodes:
        episode = paths["episode_dir"]
        try:
            result = save_episode_flow(**paths, overwrite=args.overwrite)
        except (ValueError, RuntimeError, OSError, KeyError, EOFError, zipfile.BadZipFile) as exc:
            counts["failed"] += 1
            print(f"ERROR {episode}: {exc}", file=sys.stderr)
            continue
        counts[result["status"]] += 1
        print(f"{result['status'].upper()} {episode} ({result['frame_pairs']} pairs per finger)")
    print(json.dumps(counts, sort_keys=True))
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
