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


def inspect_episode(episode_path: Path, root: Path, *, require_valid: bool = True,
                    require_three_contacts: bool = True) -> dict:
    metadata_path = episode_path / "metadata.json"
    rgb_path = episode_path / "tactile_rgb.npz"
    signals_path = episode_path / "signals.csv"
    metadata = read_json(metadata_path)
    if "collection_complete" in metadata and metadata["collection_complete"] is not True:
        raise ValueError("collection_complete is not true")
    if require_valid and metadata.get("valid") is not True:
        raise ValueError(f"Episode invalid: {metadata.get('reason', 'valid flag missing')}")
    if require_three_contacts and metadata.get("stable_three_point_contact") is not True:
        raise ValueError("No stable three-point contact")
    if metadata.get("tactile_images_saved") is not True:
        raise ValueError("tactile_images_saved is not true")
    label = metadata.get("object")
    if not isinstance(label, str) or not label.strip():
        raise ValueError("Missing non-empty metadata.object class label")
    signals = read_signals(signals_path)
    count = len(signals["time"])
    if metadata.get("num_frames") != count:
        raise ValueError(f"metadata.num_frames differs from signals length {count}")
    if not np.any(signals["phase"] == "hold"):
        raise ValueError("No hold frames available for the static branch")
    if require_three_contacts:
        stable = np.all(signals["contacts"], axis=0) & (signals["phase"] == "hold")
        if not stable.any():
            raise ValueError("Three-point flag is inconsistent with recorded hold contacts")
    shape = None
    with np.load(rgb_path, allow_pickle=False) as archive:
        if not set(FINGERS).issubset(archive.files):
            raise ValueError(f"RGB archive requires keys {FINGERS}")
        for finger in FINGERS:
            rgb = archive[finger]
            if rgb.ndim != 4 or rgb.shape[0] != count or rgb.shape[-1] != 3:
                raise ValueError(f"{finger}: expected [T,H,W,3] with T={count}; got {rgb.shape}")
            if rgb.dtype != np.uint8:
                raise ValueError(f"{finger}: RGB must be uint8, got {rgb.dtype}")
            if min(rgb.shape[1:3]) <= 0 or (shape is not None and rgb.shape != shape):
                raise ValueError("Three fingers must have identical non-empty image shapes")
            shape = rgb.shape
    depth_path = episode_path / "tactile_depth.npz"
    if depth_path.exists():
        with np.load(depth_path, allow_pickle=False) as archive:
            for finger in FINGERS:
                if finger not in archive.files:
                    raise ValueError(f"Depth archive missing {finger}")
                depth = archive[finger]
                if depth.shape != shape[:3] or not np.isfinite(depth).all():
                    raise ValueError(f"Invalid depth shape or non-finite values in {finger}")
    return {"id": episode_path.relative_to(root).as_posix(), "path": str(episode_path),
            "label": label, "metadata": metadata, "num_frames": count,
            "image_height": shape[1], "image_width": shape[2],
            "source_files": [file_signature(p) for p in [metadata_path, rgb_path, signals_path]],
            **grouping_fields(metadata)}


def build_index(root: str | Path, output: str | Path, *, require_valid: bool = True,
                require_three_contacts: bool = True) -> dict:
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f"Dataset root does not exist: {root}")
    episodes, rejected = [], []
    candidates = sorted(root.rglob("metadata.json"))
    if not candidates:
        raise ValueError(f"No metadata.json found below {root}")
    for metadata_path in candidates:
        episode_path = metadata_path.parent
        try:
            episodes.append(inspect_episode(episode_path, root, require_valid=require_valid,
                                             require_three_contacts=require_three_contacts))
        except (ValueError, OSError, KeyError, TypeError, EOFError, BadZipFile) as error:
            rejected.append({"id": episode_path.relative_to(root).as_posix(), "reason": str(error)})
    classes = sorted({episode["label"] for episode in episodes})
    for episode in episodes:
        episode["label_id"] = classes.index(episode["label"])
    result = {"schema_version": 1, "root": str(root), "classes": classes,
              "episodes": episodes, "rejected": rejected,
              "policy": {"require_valid": require_valid,
                         "require_three_contacts": require_three_contacts}}
    write_json(output, result)
    return result
