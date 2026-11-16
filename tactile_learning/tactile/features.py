"""Continuous-frame optical flow and versioned, source-checked feature caches."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Iterable

import numpy as np

FINGERS = ("finger1", "finger2", "finger3")
FEATURE_VERSION = 1


def source_fingerprint(paths: Iterable[str | Path], settings: dict | None = None) -> dict:
    files = []
    for path in sorted({str(Path(p).resolve()) for p in paths}):
        stat = Path(path).stat()
        files.append({"path": path, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    return {"files": files, "settings": settings or {}}


def cache_metadata(path: str | Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"].item()))
    if not isinstance(metadata, dict):
        raise ValueError(f"Invalid cache metadata: {path}")
    return metadata


def cache_is_valid(path: str | Path, fingerprint: dict) -> bool:
    try:
        return cache_metadata(path).get("source_fingerprint") == fingerprint
    except (OSError, ValueError, KeyError, EOFError):
        return False


def validate_cache_source(cache_path: str | Path, episode_path: str | Path, kind: str) -> dict:
    """Raise on stale or foreign caches; return metadata without loading large arrays."""
    if kind not in {"flow", "geometry"}:
        raise ValueError(f"Unknown feature kind: {kind}")
    metadata = cache_metadata(cache_path)
    if metadata.get("kind") != kind or metadata.get("schema_version") != FEATURE_VERSION:
        raise ValueError(f"Unsupported {kind} cache: {cache_path}")
    fingerprint = metadata.get("source_fingerprint", {})
    files = fingerprint.get("files", [])
    episode = Path(episode_path).resolve()
    required = {str(episode / "signals.csv")}
    if kind == "flow":
        required.add(str(episode / "tactile_rgb.npz"))
    recorded = {str(Path(item["path"]).resolve()) for item in files}
    if not required.issubset(recorded):
        raise ValueError(f"Cache belongs to another episode or has missing source files: {cache_path}")
    if kind == "geometry" and not any(Path(p).suffix.lower() == ".xml" for p in recorded):
        raise ValueError(f"Geometry cache has no source XML: {cache_path}")
    try:
        actual = source_fingerprint(recorded, fingerprint.get("settings", {}))
    except OSError as exc:
        raise ValueError(f"Cache source no longer exists: {cache_path}") from exc
    if actual != fingerprint:
        raise ValueError(f"Stale cache; rerun precomputation: {cache_path}")
    return metadata


def atomic_save_npz(path: str | Path, metadata: dict, **arrays: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
            temp_path = Path(handle.name)
            np.savez_compressed(handle, metadata=np.asarray(json.dumps(metadata, sort_keys=True)), **arrays)
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def feature_path(output: str | Path, kind: str, episode_id: str) -> Path:
    relative = PurePosixPath(episode_id)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts or ":" in episode_id or "\\" in episode_id:
        raise ValueError(f"Episode id must be a safe relative POSIX path: {episode_id!r}")
    return Path(output) / Path(*relative.parts) / f"{kind}.npz"


def read_signals(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No signal rows: {path}")
    return rows


def frame_intervals(rows: list[dict], expected_frames: int | None = None) -> np.ndarray:
    if expected_frames is not None and len(rows) != expected_frames:
        raise ValueError(f"RGB/signals frame mismatch: {expected_frames} versus {len(rows)}")
    times = np.asarray([float(row["time"]) for row in rows], dtype=np.float64)
    if len(times) < 2 or not np.isfinite(times).all() or not (np.diff(times) > 0).all():
        raise ValueError("Flow requires at least two strictly increasing finite timestamps")
    return np.diff(times).astype(np.float32)


def load_rgb(path: str | Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        arrays = [archive[key] for key in FINGERS]
    if any(a.ndim != 4 or a.shape[-1] != 3 or a.dtype != np.uint8 for a in arrays):
        raise ValueError(f"Expected each RGB stream [T,H,W,3] uint8: {path}")
    if len({a.shape for a in arrays}) != 1 or arrays[0].shape[0] < 2:
        raise ValueError(f"Three synchronized RGB streams with >=2 frames required: {path}")
    return np.stack(arrays)


def _cv2():
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("Optical-flow preprocessing requires opencv-python; install it in your selected environment") from exc
    cv2.setNumThreads(1)
    return cv2


def farneback_flow(rgb: np.ndarray, height: int = 128, width: int = 96) -> np.ndarray:
    """[3,T,H,W,3] -> normalized adjacent-frame displacement [3,T-1,2,h,w]."""
    cv2 = _cv2()
    if height < 16 or width < 16:
        raise ValueError("Flow height/width must be >=16")
    result = np.empty((rgb.shape[0], rgb.shape[1] - 1, 2, height, width), dtype=np.float16)
    for sensor in range(rgb.shape[0]):
        previous = None
        for t, frame in enumerate(rgb[sensor]):
            resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            current = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
            if previous is not None:
                flow = cv2.calcOpticalFlowFarneback(previous, current, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                flow[..., 0] /= width
                flow[..., 1] /= height
                result[sensor, t - 1] = np.moveaxis(flow, -1, 0)
            previous = current
    return result


class RaftFlow:
    """One RAFT-small model per process; CUDA uses batches without subprocesses."""

    def __init__(self, device: str = "cuda", batch_size: int = 8, iterations: int = 12):
        import torch
        from torchvision.models.optical_flow import Raft_Small_Weights, raft_small
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; select --backend farneback or a CUDA PyTorch installation")
        if batch_size < 1 or iterations < 1:
            raise ValueError("RAFT batch size and iteration count must be positive")
        self.torch = torch
        self.device = torch.device(device)
        weights = Raft_Small_Weights.DEFAULT
        self.transform = weights.transforms()
        self.model = raft_small(weights=weights, progress=True).to(self.device).eval()
        self.batch_size = batch_size
        self.iterations = iterations

    def __call__(self, rgb: np.ndarray, height: int = 128, width: int = 96) -> np.ndarray:
        torch = self.torch
        import torch.nn.functional as functional
        if height < 16 or width < 16:
            raise ValueError("Flow height/width must be >=16")
        # Resize continuous sequences once. RAFT's correlation pyramid needs >=128 pixels.
        images = torch.from_numpy(rgb).permute(0, 1, 4, 2, 3).reshape(-1, 3, *rgb.shape[2:4]).float() / 255.0
        images = functional.interpolate(images, size=(height, width), mode="bilinear", align_corners=False)
        images = images.reshape(rgb.shape[0], rgb.shape[1], 3, height, width)
        first = images[:, :-1].reshape(-1, 3, height, width)
        second = images[:, 1:].reshape(-1, 3, height, width)
        padded_h = max(128, ((height + 7) // 8) * 8)
        padded_w = max(128, ((width + 7) // 8) * 8)
        chunks = []
        with torch.inference_mode():
            for start in range(0, len(first), self.batch_size):
                a = first[start:start + self.batch_size].to(self.device)
                b = second[start:start + self.batch_size].to(self.device)
                a, b = self.transform(a, b)  # [0,1] float -> [-1,1].
                a = functional.pad(a, (0, padded_w - width, 0, padded_h - height), mode="replicate")
                b = functional.pad(b, (0, padded_w - width, 0, padded_h - height), mode="replicate")
                flow = self.model(a, b, num_flow_updates=self.iterations)[-1][..., :height, :width]
                flow[:, 0] /= width
                flow[:, 1] /= height
                chunks.append(flow.cpu().numpy().astype(np.float16))
        return np.concatenate(chunks).reshape(rgb.shape[0], rgb.shape[1] - 1, 2, height, width)


def compute_episode_flow(episode: dict, output: str | Path, config: dict, force: bool = False,
                         raft: RaftFlow | None = None) -> dict:
    folder = Path(episode["path"])
    source = folder / "tactile_rgb.npz"
    signals = folder / "signals.csv"
    fingerprint = source_fingerprint([source, signals], config)
    target = feature_path(output, "flow", episode["id"])
    if not force and cache_is_valid(target, fingerprint):
        return {"id": episode["id"], "status": "cached", "path": str(target)}
    rgb = load_rgb(source)
    dt = frame_intervals(read_signals(signals), rgb.shape[1])
    if config["backend"] == "farneback":
        flow = farneback_flow(rgb, config["height"], config["width"])
    elif config["backend"] == "raft":
        if raft is None:
            raise ValueError("RAFT backend requires a persistent RaftFlow instance")
        flow = raft(rgb, config["height"], config["width"])
    else:
        raise ValueError(f"Unknown backend: {config['backend']}")
    if not np.isfinite(flow).all():
        raise ValueError(f"Nonfinite optical flow in {episode['id']}")
    # Never bless a cache if another process edited its input while we were computing.
    if fingerprint != source_fingerprint([source, signals], config):
        raise RuntimeError(f"Source changed during preprocessing: {folder}")
    metadata = {"schema_version": FEATURE_VERSION, "kind": "flow", "source_fingerprint": fingerprint,
                "backend": config["backend"], "config": config, "units": "u/width,v/height per adjacent pair",
                "frame_count": int(rgb.shape[1]), "finger_order": list(FINGERS)}
    atomic_save_npz(target, metadata, flow=flow, dt=dt)
    return {"id": episode["id"], "status": "written", "path": str(target)}
