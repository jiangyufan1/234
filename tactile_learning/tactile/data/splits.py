"""Class-preserving, group-disjoint splits for tactile recognition experiments."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import hashlib
import json

import numpy as np

from tactile.data.index import read_json, write_json


def _group_key(episode: dict, protocol: str, block_mm: float) -> str:
    if protocol == "grouped":
        condition = episode.get("condition", {})
        required = ["quaternion_wxyz", "offset_m", "approach_azimuth_deg",
                    "approach_tilt_deg", "gripper_roll_deg"]
        if any(condition.get(key) is None for key in required):
            raise ValueError(f"{episode['id']}: missing pose_config fields needed to group repeated conditions")
        return episode["condition_key"]
    if protocol == "position":
        position = episode.get("position_m")
        if position is None:
            raise ValueError(f"{episode['id']}: grasp_offset_object_m is required for a position split")
        cell = np.floor(np.asarray(position) * 1000.0 / block_mm + 1e-9).astype(int)
        return ",".join(map(str, cell))
    if protocol == "pose":
        quaternion = episode.get("pose_quaternion_wxyz")
        if quaternion is None:
            raise ValueError(f"{episode['id']}: object_quat_wxyz is required for a pose split")
        return json.dumps(quaternion, separators=(",", ":"))
    raise ValueError(f"Unknown split protocol {protocol}")


def _partition_groups(keys: list[str], rng: np.random.Generator,
                      val_fraction: float, test_fraction: float) -> dict[str, list[str]]:
    count = len(keys)
    if count < 3:
        raise ValueError(f"Need at least 3 distinct groups per class for train/val/test; found {count}")
    keys = [keys[i] for i in rng.permutation(count)]
    n_val, n_test = max(1, round(count * val_fraction)), max(1, round(count * test_fraction))
    while n_val + n_test >= count:
        if n_test >= n_val and n_test > 1:
            n_test -= 1
        elif n_val > 1:
            n_val -= 1
        else:
            raise ValueError("Unable to allocate nonempty training groups")
    return {"test": keys[:n_test], "val": keys[n_test:n_test + n_val],
            "train": keys[n_test + n_val:]}


def _near_positions(positions: np.ndarray, reference: np.ndarray, buffer_m: float) -> np.ndarray:
    """Blockwise distances avoid allocating N x M x 3 for a dense entire dataset."""
    reference = np.unique(reference, axis=0)
    near = np.zeros(len(positions), dtype=bool)
    for begin in range(0, len(positions), 256):
        points = positions[begin:begin + 256]
        nearby = np.zeros(len(points), dtype=bool)
        for other in range(0, len(reference), 512):
            delta = points[:, None] - reference[None, other:other + 512]
            nearby |= (np.sum(delta * delta, axis=-1) <= buffer_m**2 + 1e-14).any(axis=1)
        near[begin:begin + len(points)] = nearby
    return near


def build_splits(index_path: str | Path, output: str | Path, *, protocol: str = "grouped",
                 seed: int = 42, val_fraction: float = .15, test_fraction: float = .15,
                 block_mm: float = 15.0, buffer_mm: float = 0.0,
                 allow_single_class: bool = False) -> dict:
    if val_fraction <= 0 or test_fraction <= 0 or val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction and test_fraction must be positive with sum < 1")
    if block_mm <= 0 or buffer_mm < 0:
        raise ValueError("block_mm must be positive and buffer_mm nonnegative")
    if protocol != "position" and buffer_mm:
        raise ValueError("buffer_mm applies only to the position protocol")
    index_path = Path(index_path).resolve()
    index = read_json(index_path)
    if index.get("schema_version") != 1:
        raise ValueError("Unsupported index schema")
    if len(index["classes"]) < (1 if allow_single_class else 2):
        raise ValueError("Need at least 2 object classes. Use --allow-single-class only for a pipeline diagnostic.")
    by_class = defaultdict(list)
    ids = set()
    for episode in index["episodes"]:
        if episode["id"] in ids:
            raise ValueError(f"Duplicate episode id: {episode['id']}")
        ids.add(episode["id"])
        by_class[episode["label"]].append(episode)
    partitions: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    excluded, excluded_reasons, group_assignments, class_counts = [], {}, {}, {}
    for label in index["classes"]:
        episodes = by_class[label]
        groups = defaultdict(list)
        for episode in episodes:
            groups[_group_key(episode, protocol, block_mm)].append(episode)
        # Adding unrelated classes does not change this class's random split.
        label_seed = int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "little")
        rng = np.random.default_rng((seed + label_seed) % (2**64))
        try:
            selected = _partition_groups(sorted(groups), rng, val_fraction, test_fraction)
        except ValueError as error:
            raise ValueError(f"Class {label!r}, protocol {protocol}: {error}. "
                             "Collect more conditions or select a different protocol; no episode-level fallback is used.") from error
        per_class = {part: [e for key in keys for e in groups[key]] for part, keys in selected.items()}
        group_assignments[label] = selected
        if protocol == "position" and buffer_mm > 0:
            # Same object frame and class only. Repeats at an offset are removed together.
            for source, heldout in [("val", "test"), ("train", "test"), ("train", "val")]:
                records, reference = per_class[source], per_class[heldout]
                if not records or not reference:
                    continue
                near = _near_positions(np.asarray([e["position_m"] for e in records]),
                                       np.asarray([e["position_m"] for e in reference]), buffer_mm / 1000)
                per_class[source] = [e for e, removed in zip(records, near) if not removed]
                for episode, removed in zip(records, near):
                    if removed:
                        excluded.append(episode["id"])
                        excluded_reasons[episode["id"]] = f"within {buffer_mm:g} mm of {heldout} target positions"
        counts = {part: len(records) for part, records in per_class.items()}
        if not all(counts.values()):
            raise ValueError(f"Class {label!r} has an empty partition after buffering: {counts}. "
                             "Increase spatial block size, reduce buffer, or collect more distinct regions.")
        class_counts[label] = counts
        for part, records in per_class.items():
            partitions[part].extend(e["id"] for e in records)
    result = {"schema_version": 1, "index_path": str(index_path), "seed": seed,
              "protocol": protocol, "index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
              "fractions": {"train": 1-val_fraction-test_fraction, "val": val_fraction, "test": test_fraction},
              "position_block_mm": block_mm if protocol == "position" else None,
              "position_buffer_mm": buffer_mm, "group_assignments": group_assignments,
              "class_counts": class_counts, "excluded": sorted(excluded),
              "excluded_reasons": excluded_reasons,
              "notes": "Ratios apply approximately to groups. Pose groups use commanded quaternions, not pose_group names.",
              **{part: sorted(values) for part, values in partitions.items()}}
    validate_split(index, result)
    write_json(output, result)
    return result


def validate_split(index: dict, split: dict) -> None:
    """Reject overlapping sets and stale/unknown IDs on every dataset load."""
    if split.get("schema_version") != 1:
        raise ValueError("Unsupported split schema")
    known = {record["id"] for record in index["episodes"]}
    seen = set()
    for part in ["train", "val", "test", "excluded"]:
        values = split.get(part)
        if not isinstance(values, list):
            raise ValueError(f"Split field {part} must be a list")
        current = set(values)
        if len(current) != len(values) or seen.intersection(current):
            raise ValueError(f"Duplicate or overlapping episode IDs in {part}")
        if not current.issubset(known):
            raise ValueError(f"Unknown episode IDs in {part}: {sorted(current-known)[:3]}")
        seen.update(current)
    if seen != known:
        raise ValueError("Split does not account for every indexed episode")
    if split.get("protocol") == "debug":
        if split["val"] or split["test"] or split["excluded"]:
            raise ValueError("Debug splits must place every episode in train only")
        return
    # Protect against hand-edited split files that separate repeated conditions.
    assignment = {}
    for part in ["train", "val", "test"]:
        current = set(split[part])
        for episode in index["episodes"]:
            if episode["id"] not in current:
                continue
            key = (episode["label"], _group_key(episode, split["protocol"],
                                               split.get("position_block_mm") or 15.0))
            if key in assignment and assignment[key] != part:
                raise ValueError(f"Group leakage across partitions for {episode['id']}")
            assignment[key] = part
