"""Episode-level tensors with explicit padding/contact masks.

``contact_source='signals'`` uses simulator contact flags as an oracle mask.
Use ``'all'`` for a no-contact-oracle ablation; object pose/position and class
metadata never become input features. Geometry is reconstructed separately
from gripper kinematics, never copied from ``gel*_obj_*`` CSV fields.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from tactile.data.index import FINGERS, read_json, read_signals, validate_signatures
from tactile.data.splits import validate_split


def _sample_indices(available: np.ndarray, maximum: int) -> np.ndarray:
    if len(available) <= maximum:
        return available
    return available[np.linspace(0, len(available) - 1, maximum).round().astype(int)]


class TactileEpisodeDataset(Dataset):
    """One episode -> synchronized RGB, cached flow, kinematics and label.

    RGB and flow retain a fixed finger axis of length three. Invalid/padded
    slots have false masks. Flow is displacement normalized by image width
    and height; resizing a cached flow field does not change its magnitude.
    Optical-flow sampling retains original adjacent-frame pairs and their
    actual pair duration, even when some pairs are omitted to cap sequence
    length. The three fingers always use the same temporal indices.
    """

    def __init__(self, index_path: str | Path, split_path: str | Path,
                 partition: str, config: dict):
        self.index_path, self.split_path = Path(index_path).resolve(), Path(split_path).resolve()
        index, split = read_json(self.index_path), read_json(self.split_path)
        if index.get("schema_version") != 1:
            raise ValueError("Unsupported index schema")
        if Path(split["index_path"]).resolve() != self.index_path:
            raise ValueError("Split was created for a different index_path")
        if split.get("index_sha256") and split["index_sha256"] != hashlib.sha256(self.index_path.read_bytes()).hexdigest():
            raise ValueError("Index changed after splitting; rebuild the splits")
        validate_split(index, split)
        if partition not in ["train", "val", "test"]:
            raise ValueError(f"Unknown dataset partition {partition}")
        self.episode_ids = list(split[partition])
        if not self.episode_ids:
            raise ValueError(f"The {partition} partition is empty")
        by_id = {record["id"]: record for record in index["episodes"]}
        self.episodes = [by_id[key] for key in self.episode_ids]
        self.labels = [record["label_id"] for record in self.episodes]
        self.classes = list(index["classes"])
        if any(record["label"] != self.classes[record["label_id"]] for record in self.episodes):
            raise ValueError("Index labels do not match the saved class mapping")
        self.config = dict(config)
        self.height = int(config.get("height", 128))
        self.width = int(config.get("width", 96))
        self.static_frames = int(config.get("static_frames", 4))
        self.max_flow_frames = int(config.get("max_flow_frames", 31))
        if min(self.height, self.width, self.static_frames, self.max_flow_frames) <= 0:
            raise ValueError("Image dimensions and temporal lengths must be positive")
        self.use_flow = bool(config.get("use_flow", False))
        self.use_geometry = bool(config.get("use_geometry", False))
        self.normalize_rgb = bool(config.get("normalize_rgb", True))
        self.contact_source = config.get("contact_source", "signals")
        if self.contact_source not in ["signals", "all"]:
            raise ValueError("contact_source must be 'signals' or 'all'")
        self.flow_cache = self._cache_root(config, "flow_cache", self.use_flow)
        self.geometry_cache = self._cache_root(config, "geometry_cache", self.use_geometry)
        self.mean = torch.tensor([.485, .456, .406]).view(1, 1, 3, 1, 1)
        self.std = torch.tensor([.229, .224, .225]).view(1, 1, 3, 1, 1)

    @staticmethod
    def _cache_root(config: dict, key: str, required: bool) -> Path | None:
        value = config.get(key)
        if required and not value:
            raise ValueError(f"{key} must be set when its feature is enabled")
        return Path(value).resolve() if value else None

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, index: int) -> dict:
        episode = self.episodes[index]
        try:
            return self._load(episode)
        except (ValueError, OSError, KeyError, RuntimeError) as error:
            raise ValueError(f"Episode {episode['id']}: {error}") from error

    def _load(self, episode: dict) -> dict:
        path = Path(episode["path"])
        if not episode.get("source_files"):
            raise ValueError("Index lacks source fingerprints; rebuild it with build_dataset_index.py")
        validate_signatures(episode["source_files"])
        signals = read_signals(path / "signals.csv")
        times = signals["time"]
        count = len(times)
        contact = signals["contacts"] if self.contact_source == "signals" else np.ones((3, count), dtype=bool)
        hold = np.flatnonzero((signals["phase"] == "hold") & contact.any(axis=0))
        static_indices = _sample_indices(hold, self.static_frames)
        rgb = torch.zeros(3, self.static_frames, 3, self.height, self.width)
        rgb_mask = torch.zeros(3, self.static_frames, dtype=torch.bool)
        if len(static_indices):
            with np.load(path / "tactile_rgb.npz", allow_pickle=False) as archive:
                images = np.stack([archive[finger][static_indices] for finger in FINGERS])
            image_tensor = torch.from_numpy(images).permute(0, 1, 4, 2, 3).float() / 255.0
            image_tensor = F.interpolate(image_tensor.flatten(0, 1), size=(self.height, self.width),
                                         mode="bilinear", align_corners=False, antialias=True)
            rgb[:, :len(static_indices)] = image_tensor.reshape(3, len(static_indices), 3, self.height, self.width)
            rgb_mask[:, :len(static_indices)] = torch.from_numpy(contact[:, static_indices].copy())
        if self.normalize_rgb:
            rgb = (rgb - self.mean) / self.std
        rgb *= rgb_mask[:, :, None, None, None]
        flow = torch.zeros(3, self.max_flow_frames, 2, self.height, self.width)
        flow_mask = torch.zeros(3, self.max_flow_frames, dtype=torch.bool)
        flow_dt = torch.zeros(self.max_flow_frames)
        if self.use_flow:
            from tactile.features import validate_cache_source
            cache_path = self.flow_cache / episode["id"] / "flow.npz"
            cache_info = validate_cache_source(cache_path, path, "flow")
            if cache_info.get("backend") != self.config.get("flow_backend", "farneback"):
                raise ValueError("Flow backend differs from data.flow_backend; do not mix Farneback and RAFT caches")
            if cache_info.get("units") != "u/width,v/height per adjacent pair" or cache_info.get("finger_order") != list(FINGERS):
                raise ValueError("Flow cache must use normalized displacement and the standard finger order")
            with np.load(cache_path, allow_pickle=False) as archive:
                cached_flow = archive["flow"]
                cached_dt = archive["dt"]
            if cached_flow.ndim != 5 or cached_flow.shape[:3] != (3, count - 1, 2):
                raise ValueError(f"Expected flow [3,{count-1},2,H,W], got {cached_flow.shape}")
            if cached_dt.shape != (count - 1,) or not np.isfinite(cached_flow).all() or not np.isfinite(cached_dt).all():
                raise ValueError("Non-finite or incorrectly shaped flow cache")
            if not np.allclose(cached_dt, np.diff(times), rtol=1e-4, atol=1e-7):
                raise ValueError("Flow dt does not match signals timestamps")
            flow_indices = _sample_indices(np.arange(count - 1), self.max_flow_frames)
            tensor = torch.from_numpy(cached_flow[:, flow_indices].astype(np.float32, copy=False))
            tensor = F.interpolate(tensor.flatten(0, 1), size=(self.height, self.width),
                                   mode="bilinear", align_corners=False)
            flow[:, :len(flow_indices)] = tensor.reshape(3, len(flow_indices), 2, self.height, self.width)
            pair_mask = contact[:, flow_indices] | contact[:, flow_indices + 1]
            flow_mask[:, :len(flow_indices)] = torch.from_numpy(pair_mask.copy())
            flow_dt[:len(flow_indices)] = torch.from_numpy(cached_dt[flow_indices].astype(np.float32))
            flow *= flow_mask[:, :, None, None, None]
        sensor_pose = torch.zeros(3, 12)
        edge_attr = torch.zeros(3, 3, 5)
        if self.use_geometry:
            from tactile.features import validate_cache_source
            cache_path = self.geometry_cache / episode["id"] / "geometry.npz"
            cache_info = validate_cache_source(cache_path, path, "geometry")
            settings = cache_info.get("config", {})
            if settings.get("reference_body") != self.config.get("geometry_reference_body", "gripper_mount"):
                raise ValueError("Geometry reference body differs from data.geometry_reference_body")
            expected = {"position_unit": "metre", "rotation_order": "row-major",
                        "edge_direction": "i_to_j", "normal_axis": "+Z"}
            if (not settings.get("reference_body") or any(settings.get(k) != v for k, v in expected.items())
                    or cache_info.get("finger_order") != list(FINGERS)):
                raise ValueError("Geometry cache is missing its gripper reference frame or coordinate convention")
            with np.load(cache_path, allow_pickle=False) as archive:
                poses, edges = archive["sensor_pose"], archive["edge_attr"]
            if poses.shape != (count, 3, 12) or edges.shape != (count, 3, 3, 5):
                raise ValueError("Geometry cache shape must be [T,3,12] and [T,3,3,5]")
            if not np.isfinite(poses).all() or not np.isfinite(edges).all():
                raise ValueError("Non-finite geometry cache")
            # Pick one real pose, not an arithmetic mean of rotation matrices.
            hold_all = np.flatnonzero(signals["phase"] == "hold")
            if not len(hold_all):
                raise ValueError("No hold frame for representative sensor geometry")
            representative = int(hold_all[len(hold_all) // 2])
            sensor_pose = torch.from_numpy(poses[representative].astype(np.float32, copy=True))
            edge_attr = torch.from_numpy(edges[representative].astype(np.float32, copy=True))
        node_mask = rgb_mask.any(dim=1) | flow_mask.any(dim=1)
        if not node_mask.any():
            raise ValueError("No observed contact frames. Change contact_source only for a declared no-oracle experiment.")
        return {"rgb": rgb, "rgb_mask": rgb_mask, "flow": flow, "flow_mask": flow_mask,
                "flow_dt": flow_dt, "sensor_pose": sensor_pose, "edge_attr": edge_attr,
                "node_mask": node_mask, "label": torch.tensor(episode["label_id"], dtype=torch.long),
                "episode_id": episode["id"]}
