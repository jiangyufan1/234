"""Inspect episode files before training; invalid episodes never enter the index.

The metadata is used for labels and split construction, not as network input.
Legacy episodes without ``collection_complete`` are accepted only after all
saved RGB and signal records have passed validation.
"""
from __future__ import annotations

import csv
import hashlib
import json
from zipfile import BadZipFile
from pathlib import Path
from typing import Any

import numpy as np

FINGERS = ("finger1", "finger2", "finger3")
SIGNAL_COLUMNS = ("frame", "time", "phase", "q_f1", "q_f2", "q_f3",
                  "contact_f1", "contact_f2", "contact_f3")


def read_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def write_json(path: str | Path, value: Any) -> None:
    """Write atomically so an interrupted indexing job leaves no partial JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def file_signature(path: str | Path) -> dict:
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def validate_signatures(signatures: list[dict]) -> None:
    for expected in signatures:
        path = Path(expected["path"])
        if not path.is_file() or file_signature(path) != expected:
            raise ValueError(f"Source changed since indexing: {path}. Rebuild the index and splits.")


def read_signals(path: str | Path) -> dict[str, np.ndarray]:
    """Parse CSV without pandas and validate synchronized, increasing timestamps."""
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(SIGNAL_COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"signals.csv missing columns: {sorted(missing)}")
        rows = list(reader)
    if len(rows) < 2:
        raise ValueError("At least two synchronized signal frames are required")
    numeric: dict[str, np.ndarray] = {}
    for key in reader.fieldnames or []:
        if key == "phase":
            continue
        try:
            values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        except (ValueError, TypeError, KeyError) as error:
            raise ValueError(f"Invalid numeric signals column {key}") from error
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite signals column {key}")
        numeric[key] = values
    frames = numeric["frame"]
    if not np.array_equal(frames, np.arange(len(rows))):
        raise ValueError("Frame indices must start at 0 and be consecutive")
    times = numeric["time"]
    if not np.all(np.diff(times) > 0):
        raise ValueError("Signal timestamps must strictly increase")
    phases = np.asarray([row["phase"] for row in rows])
    if np.any(~np.isin(phases, ["precontact", "dynamic_contact", "hold"])):
        raise ValueError(f"Unknown capture phase(s): {sorted(set(phases))}")
    contacts = np.stack([numeric[f"contact_f{i}"] for i in range(1, 4)], axis=0)
    if not np.isin(contacts, [0, 1]).all():
        raise ValueError("Contact flags must be 0 or 1")
    return {"time": times, "phase": phases, "contacts": contacts.astype(bool),
            "q": np.stack([numeric[f"q_f{i}"] for i in range(1, 4)], axis=1)}


def canonical_quaternion(value: Any) -> list[float] | None:
    if value is None:
        return None
    quaternion = np.asarray(value, dtype=np.float64)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ValueError("object_quat_wxyz must contain four finite numbers")
    norm = np.linalg.norm(quaternion)
    if norm < 1e-10:
        raise ValueError("Zero quaternion is invalid")
    quaternion /= norm
    # q and -q describe the same rotation, including 180-degree rotations.
    nonzero = np.flatnonzero(np.abs(quaternion) > 1e-8)
    if quaternion[nonzero[0]] < 0:
        quaternion = -quaternion
    return np.round(quaternion, 6).tolist()


def grouping_fields(metadata: dict) -> dict:
    pose = metadata.get("pose_config", {})
    if not isinstance(pose, dict):
        raise ValueError("pose_config must be an object")
    offset = pose.get("grasp_offset_object_m")
    if offset is not None:
        offset = np.asarray(offset, dtype=np.float64)
        if offset.shape != (3,) or not np.isfinite(offset).all():
            raise ValueError("grasp_offset_object_m must contain three finite numbers")
        offset = np.round(offset, 7).tolist()
    quaternion = canonical_quaternion(pose.get("object_quat_wxyz"))
    angles = {}
    for key in ["approach_azimuth_deg", "approach_tilt_deg", "gripper_roll_deg"]:
        value = pose.get(key)
        if value is not None:
            value = float(value)
            if not np.isfinite(value):
                raise ValueError(f"Non-finite pose_config.{key}")
            value = round(value % 360.0, 6)
        angles[key] = value
    condition = {"quaternion_wxyz": quaternion, "offset_m": offset, **angles}
    condition_key = hashlib.sha256(json.dumps(condition, sort_keys=True).encode()).hexdigest()[:24]
    return {"condition_key": condition_key, "position_m": offset,
            "pose_quaternion_wxyz": quaternion, "condition": condition}




def resolve_episode_paths(episode_path: Path, root: Path,
                          metadata: dict) -> dict:
    """Resolve legacy or five-directory (by_type) episode storage."""

    storage = metadata.get("storage", {})
    if not isinstance(storage, dict):
        raise ValueError("metadata.storage must be an object")

    mapping = storage.get("run_relative_paths", {})
    if isinstance(mapping, str):
        try:
            mapping = json.loads(mapping)
        except json.JSONDecodeError as error:
            raise ValueError("Invalid run_relative_paths JSON") from error

    if not isinstance(mapping, dict):
        raise ValueError("run_relative_paths must be an object")

    if storage.get("layout") != "by_type":
        return {
            "layout": "legacy",
            "metadata": episode_path / "metadata.json",
            "rgb": episode_path / "tactile_rgb.npz",
            "signals": episode_path / "signals.csv",
            "flow": None,
            "depth": episode_path / "tactile_depth.npz",
        }

    required = ("metadata", "rgb", "flow", "signals")
    missing = [key for key in required if not mapping.get(key)]
    if missing:
        raise ValueError(f"Missing by_type paths: {missing}")

    def safe_path(relative):
        path = (root / relative).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError as error:
            raise ValueError(
                f"Storage path escapes dataset root: {relative}"
            ) from error
        return path

    paths = {
        "layout": "by_type",
        "metadata": safe_path(mapping["metadata"]),
        "rgb": safe_path(mapping["rgb"]),
        "flow": safe_path(mapping["flow"]),
        "signals": safe_path(mapping["signals"]),
        "depth": safe_path(mapping["depth"]) if mapping.get("depth") else None,
    }

    if paths["metadata"] != (episode_path / "metadata.json").resolve():
        raise ValueError("Metadata path does not match scanned file")

    return paths


def _png_dimensions(path: Path) -> tuple[int, int]:
    """Read PNG dimensions from its header without decoding image pixels."""
    import struct

    with path.open("rb") as handle:
        header = handle.read(24)

    if len(header) != 24 or header[:8] != b"\\x89PNG\\r\\n\\x1a\\n":
        raise ValueError(f"Invalid PNG file: {path}")

    if header[12:16] != b"IHDR":
        raise ValueError(f"Missing PNG IHDR: {path}")

    return struct.unpack(">II", header[16:24])


def validate_by_type_flow(path: Path, expected_pairs: int) -> None:
    """Check the existing flow archive without decompressing large flow arrays."""
    if not path.is_file():
        raise ValueError(f"Missing flow archive: {path}")

    try:
        with np.load(path, allow_pickle=False) as data:
            required = {
                "frame_from", "frame_to", "time_start", "time_end", "dt",
                "contact_start", "contact_end", "contact_pair",
                "finger1", "finger2", "finger3", "units", "metadata",
            }

            missing = required - set(data.files)
            if missing:
                raise ValueError(f"Flow archive missing members: {sorted(missing)}")

            units = str(data["units"].item())
            if units != "pixels/frame_pair":
                raise ValueError(f"Unexpected flow units: {units}")

            if data["frame_from"].shape != (expected_pairs,):
                raise ValueError("Incorrect flow frame_from length")

            if data["frame_to"].shape != (expected_pairs,):
                raise ValueError("Incorrect flow frame_to length")

            if data["dt"].shape != (expected_pairs,):
                raise ValueError("Incorrect flow dt length")

            if data["contact_pair"].shape != (expected_pairs, 3):
                raise ValueError("Incorrect flow contact_pair shape")

            if not np.isfinite(data["dt"]).all() or not np.all(data["dt"] > 0):
                raise ValueError("Flow dt must be finite and positive")

    except (OSError, EOFError, BadZipFile) as error:
        raise ValueError(f"Unreadable flow archive: {path}") from error



def inspect_episode(episode_path: Path, root: Path, *,
                    require_valid: bool = True,
                    require_three_contacts: bool = False) -> dict:
    """Validate one legacy or by_type episode."""

    metadata_path = episode_path / "metadata.json"
    metadata = read_json(metadata_path)
    paths = resolve_episode_paths(episode_path, root, metadata)

    complete = metadata.get("collection_complete")
    if complete is False:
        raise ValueError("collection_complete is false")
    if paths["layout"] == "by_type" and complete is not True:
        raise ValueError("by_type requires collection_complete=True")

    if require_valid and metadata.get("valid") is not True:
        raise ValueError(
            f"Episode invalid: {metadata.get('reason', 'valid flag missing')}"
        )

    if require_three_contacts and metadata.get(
        "stable_three_point_contact"
    ) is not True:
        raise ValueError("No stable three-point contact")

    if metadata.get("tactile_images_saved") is not True:
        raise ValueError("tactile_images_saved is not true")

    label = metadata.get("object")
    if not isinstance(label, str) or not label.strip():
        raise ValueError("Missing non-empty metadata.object class label")

    signals = read_signals(paths["signals"])
    count = len(signals["time"])

    if metadata.get("num_frames") != count:
        raise ValueError("Metadata and Signals frame counts differ")

    if not np.any(signals["phase"] == "hold"):
        raise ValueError("No hold frames available for the static branch")

    if require_three_contacts:
        stable = (
            np.all(signals["contacts"], axis=0)
            & (signals["phase"] == "hold")
        )
        if not stable.any():
            raise ValueError("No stable three-point hold contact")

    source_paths = [metadata_path, paths["signals"]]

    if paths["layout"] == "by_type":
        if metadata.get("optical_flow_saved") is not True:
            raise ValueError("optical_flow_saved is not true")

        rgb_files = {}
        dimensions = None

        for finger in FINGERS:
            folder = paths["rgb"] / finger
            if not folder.is_dir():
                raise ValueError(f"Missing RGB directory: {folder}")

            files = sorted(folder.glob("frame_*.png"))
            expected = [
                f"frame_{i:04d}.png" for i in range(count)
            ]

            if [p.name for p in files] != expected:
                raise ValueError(
                    f"{finger}: expected {count} consecutive PNG frames, "
                    f"found {len(files)}"
                )

            current_dimensions = _png_dimensions(files[0])
            if dimensions is not None and current_dimensions != dimensions:
                raise ValueError("Three finger RGB dimensions differ")

            dimensions = current_dimensions
            rgb_files[finger] = [str(p.resolve()) for p in files]

        validate_by_type_flow(paths["flow"], count - 1)

        source_paths.append(paths["flow"])

        if paths["depth"] is not None and paths["depth"].is_file():
            depth_available = True
            source_paths.append(paths["depth"])
        else:
            depth_available = False

        width, height = dimensions

        extra = {
            "storage_layout": "by_type",
            "depth_available": depth_available,
            "paths": {
                "metadata": str(metadata_path.resolve()),
                "rgb_dir": str(paths["rgb"].resolve()),
                "flow": str(paths["flow"].resolve()),
                "signals": str(paths["signals"].resolve()),
                "depth": (
                    str(paths["depth"].resolve())
                    if depth_available else None
                ),
            },
            "rgb_files": rgb_files,
        }

    else:
        rgb_path = paths["rgb"]

        if not rgb_path.is_file():
            raise ValueError(f"Missing RGB archive: {rgb_path}")

        shape = None

        with np.load(rgb_path, allow_pickle=False) as archive:
            if not set(FINGERS).issubset(archive.files):
                raise ValueError(f"RGB archive requires keys {FINGERS}")

            for finger in FINGERS:
                rgb = archive[finger]

                if rgb.ndim != 4 or rgb.shape[0] != count or rgb.shape[-1] != 3:
                    raise ValueError(
    f"{finger}: expected [T,H,W,3] with T={count}; got {rgb.shape}"
)

                if rgb.dtype != np.uint8:
                    raise ValueError(f"Invalid RGB dtype: {finger}")

                if shape is not None and rgb.shape != shape:
                    raise ValueError("Three finger RGB shapes differ")

                shape = rgb.shape

        height, width = shape[1:3]
        source_paths.append(rgb_path)

        depth_available = (
            paths["depth"] is not None and paths["depth"].is_file()
        )

        if depth_available:
            source_paths.append(paths["depth"])

        extra = {
            "storage_layout": "legacy",
            "depth_available": depth_available,
            "paths": {
                "metadata": str(metadata_path.resolve()),
                "rgb_archive": str(rgb_path.resolve()),
                "signals": str(paths["signals"].resolve()),
                "depth": (
                    str(paths["depth"].resolve())
                    if depth_available else None
                ),
            },
        }

    return {
        "id": (
            episode_path.relative_to(root / "metadata_json").as_posix()
            if paths["layout"] == "by_type"
            else episode_path.relative_to(root).as_posix()
        ),
        "path": str(episode_path.resolve()),
        "label": label,
        "metadata": metadata,
        "num_frames": count,
        "image_height": height,
        "image_width": width,
        "source_files": [file_signature(p) for p in source_paths],
        **extra,
        **grouping_fields(metadata),
    }


def build_index(root: str | Path, output: str | Path, *,
                require_valid: bool = True,
                require_three_contacts: bool = False) -> dict:
    """Build a legacy or by_type index while preserving schema version 1."""

    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f"Dataset root does not exist: {root}")

    metadata_root = root / "metadata_json"
    by_type = metadata_root.is_dir()

    candidates = (
        sorted(metadata_root.glob("*/*/metadata.json"))
        if by_type
        else sorted(root.rglob("metadata.json"))
    )

    if not candidates:
        raise ValueError(f"No metadata.json found below {root}")

    episodes, rejected = [], []

    for number, metadata_path in enumerate(candidates, 1):
        episode_path = metadata_path.parent

        try:
            episode = inspect_episode(
                episode_path,
                root,
                require_valid=require_valid,
                require_three_contacts=require_three_contacts,
            )
            episodes.append(episode)

        except (ValueError, OSError, KeyError, TypeError, EOFError, BadZipFile) as error:
            relative_root = metadata_root if by_type else root
            rejected.append({
                "id": episode_path.relative_to(relative_root).as_posix(),
                "reason": str(error),
            })

        if number % 1000 == 0:
            print(
                f"Indexed {number}/{len(candidates)} | "
                f"accepted={len(episodes)} | rejected={len(rejected)}",
                flush=True,
            )

    classes = sorted({episode["label"] for episode in episodes})
    class_to_id = {label: i for i, label in enumerate(classes)}

    for episode in episodes:
        episode["label_id"] = class_to_id[episode["label"]]

    result = {
        "schema_version": 1,
        "root": str(root),
        "layout": "by_type" if by_type else "legacy",
        "classes": classes,
        "episodes": episodes,
        "rejected": rejected,
        "policy": {
            "require_valid": require_valid,
            "require_three_contacts": require_three_contacts,
            "require_collection_complete": by_type,
        },
    }

    write_json(output, result)

    print("\n===== INDEX SUMMARY =====")
    print("Layout:", result["layout"])
    print("Accepted:", len(episodes))
    print("Rejected:", len(rejected))
    print("Classes:", len(classes))

    return result
