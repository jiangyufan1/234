"""Paths and lossless PNG I/O for collection output; no simulator imports."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np


LAYOUTS = ("legacy", "by_type", "by_episode")
FINGERS = ("finger1", "finger2", "finger3")


def _inside(path: Path, run_root: Path) -> Path:
    """Resolve before any write/delete, rejecting traversal and link escapes."""
    if ".." in path.parts:
        raise ValueError(f"Output paths cannot contain '..': {path}")
    resolved = path.resolve()
    if resolved == run_root or not resolved.is_relative_to(run_root):
        raise ValueError(f"Output must be below run root {run_root}: {path}")
    # A symlink to a different sample within the run must also stay untouched.
    candidate = path.absolute()
    while candidate != candidate.parent:
        if candidate.resolve() == run_root:
            break
        if candidate.is_symlink() or (hasattr(candidate, "is_junction") and candidate.is_junction()):
            raise ValueError(f"Linked output directories are not supported: {candidate}")
        candidate = candidate.parent
    return resolved


def episode_paths(episode_dir, run_root, layout="legacy") -> dict[str, Path]:
    """Map a logical object/episode to its physical output paths."""
    if layout not in LAYOUTS:
        raise ValueError(f"Unknown collection output layout: {layout}")
    root = Path(run_root).resolve()
    episode = _inside(Path(episode_dir), root)
    relative = episode.relative_to(root)
    if layout == "legacy":
        paths = {
            "rgb": episode / "tactile_rgb.npz",
            "rgb_npz": episode / "tactile_rgb.npz",
            "depth": episode / "tactile_depth.npz",
            "flow": episode / "tactile_flow.npz",
            "signals": episode / "signals.csv",
            "metadata": episode / "metadata.json",
        }
    else:
        def format_root(name):
            return root / name / relative if layout == "by_type" else episode / name

        paths = {
            "rgb": format_root("rgb_png"),
            "rgb_npz": format_root("rgb_npz") / "tactile_rgb.npz",
            "depth": format_root("depth_npz") / "tactile_depth.npz",
            "flow": format_root("flow_npz") / "tactile_flow.npz",
            "signals": format_root("signals_csv") / "signals.csv",
            "metadata": format_root("metadata_json") / "metadata.json",
        }
    return {name: _inside(path, root) for name, path in paths.items()}


def storage_metadata(episode_dir, run_root, layout="legacy") -> dict:
    root = Path(run_root).resolve()
    paths = episode_paths(episode_dir, root, layout)
    relative = {
        key: paths[key].relative_to(root).as_posix()
        for key in ("rgb", "depth", "flow", "signals", "metadata")
    }
    return {"layout": layout, "run_relative_paths": json.dumps(relative, ensure_ascii=False)}


def clear_episode_files(episode_dir, run_root, layout="legacy") -> None:
    """Remove only this sample's known outputs, keeping metadata incomplete."""
    root = Path(run_root).resolve()
    paths = episode_paths(episode_dir, root, layout)
    if layout != "legacy" and paths["rgb"].exists():
        rgb = _inside(paths["rgb"], root)
        expected = episode_paths(episode_dir, root, layout)["rgb"]
        if rgb != expected or not rgb.is_dir():
            raise ValueError(f"Refusing to remove unexpected RGB output: {rgb}")
        shutil.rmtree(rgb)
    keys = ("rgb_npz", "depth", "flow", "signals") if layout == "legacy" else ("depth", "flow", "signals")
    for key in keys:
        _inside(paths[key], root).unlink(missing_ok=True)


def save_png_sequences(directory, sequences) -> None:
    """Write uint8 RGB sequences as lossless PNG, including Unicode paths."""
    import cv2

    directory = Path(directory)
    for finger in FINGERS:
        frames = np.asarray(sequences[finger])
        if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(f"{finger}: RGB must be uint8 [T,H,W,3].")
        target = directory / finger
        target.mkdir(parents=True, exist_ok=True)
        for index, frame in enumerate(frames):
            ok, encoded = cv2.imencode(".png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if not ok:
                raise OSError(f"Cannot encode PNG for {finger} frame {index}.")
            (target / f"frame_{index:04d}.png").write_bytes(encoded.tobytes())


def load_png_sequences(directory, expected_frames) -> dict[str, np.ndarray]:
    """Validate a saved sample without altering any outputs."""
    import cv2

    directory = Path(directory)
    sequences = {}
    expected_names = [f"frame_{index:04d}.png" for index in range(expected_frames)]
    for finger in FINGERS:
        folder = directory / finger
        if sorted(path.name for path in folder.glob("*.png")) != expected_names:
            raise ValueError(f"{folder}: missing, extra, or noncontiguous PNG frames.")
        frames = []
        for name in expected_names:
            encoded = np.frombuffer((folder / name).read_bytes(), dtype=np.uint8)
            frame = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
            if frame is None or frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] != 3:
                raise ValueError(f"Invalid RGB PNG: {folder / name}")
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        sequences[finger] = np.stack(frames)
    return sequences
