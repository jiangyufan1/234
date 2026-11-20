#!/usr/bin/env python3
"""Memory-bounded Touch and Go GelSight preprocessing.

- RGB: uint8 [T,H,W,3]
- Farneback flow: float16 [T-1,2,H,W], normalized by width/height
- Per-recording temporary .npy memmaps prevent whole-video arrays in RAM.
- Completed NPZs are skipped on rerun. An interrupted current recording is
  restarted from frame zero; completed recordings are retained.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np


def find_recordings(root: Path) -> list[Path]:
    return sorted({p.parent.resolve() for p in root.rglob("gelsight.mp4")})


def load_times(path: Path) -> np.ndarray:
    values = np.asarray(np.load(path, allow_pickle=True)).reshape(-1)
    if values.size < 2:
        raise ValueError(f"Need at least two timestamps: {path}")
    if np.issubdtype(values.dtype, np.number):
        times = values.astype(np.float64)
    else:
        try:
            times = values.astype("datetime64[ns]").astype("int64").astype(np.float64) / 1e9
        except (TypeError, ValueError):
            times = values.astype(np.float64)
    if not np.isfinite(times).all():
        raise ValueError(f"Non-finite timestamps: {path}")
    times -= times[0]
    if not np.all(np.diff(times) > 0):
        raise ValueError(f"Timestamps are not strictly increasing: {path}")
    return times


def annotation_preview(root: Path, limit: int = 12) -> dict:
    result = {}
    for name in ("label.txt", "category_reference.txt"):
        matches = list(root.rglob(name))
        if not matches:
            result[name] = {"found": False}
            continue
        path = matches[0]
        lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
        result[name] = {
            "found": True,
            "path": str(path),
            "line_count": len(lines),
            "first_lines": lines[:limit],
        }
    return result


def audit(root: Path, report: Path | None) -> None:
    import cv2

    rows = []
    for folder in find_recordings(root):
        video, time_file = folder / "gelsight.mp4", folder / "time2.npy"
        cap = cv2.VideoCapture(str(video))
        opened = cap.isOpened()
        info = {
            "id": folder.relative_to(root).as_posix(),
            "video_bytes": video.stat().st_size,
            "time2_exists": time_file.is_file(),
            "video_opened": opened,
            "reported_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if opened else None,
            "fps": float(cap.get(cv2.CAP_PROP_FPS)) if opened else None,
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if opened else None,
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if opened else None,
        }
        cap.release()
        if time_file.is_file():
            try:
                t = load_times(time_file)
                info.update(
                    timestamp_count=len(t),
                    dt_median=float(np.median(np.diff(t))),
                    dt_min=float(np.min(np.diff(t))),
                    dt_max=float(np.max(np.diff(t))),
                )
                info["count_match"] = info["reported_frames"] == len(t)
            except Exception as exc:
                info["timestamp_error"] = repr(exc)
        rows.append(info)

    result = {
        "root": str(root.resolve()),
        "recording_count": len(rows),
        "recordings": rows,
        "annotations": annotation_preview(root),
    }
    if report:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    valid = sum(
        bool(r.get("video_opened") and r.get("time2_exists") and r.get("count_match"))
        for r in rows
    )
    print(f"Recording folders: {len(rows)}")
    print(f"Video/timestamp count matched: {valid}")
    print(f"Needs inspection: {len(rows) - valid}")
    print("Annotation preview:")
    print(json.dumps(result["annotations"], ensure_ascii=False, indent=2))
    if report:
        print(f"Audit report: {report}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_compressed_npz(target: Path, rgb_map: np.memmap, flow_map: np.memmap,
                          times: np.ndarray, dt: np.ndarray, metadata: dict) -> None:
    """Compress memmaps into NPZ; NumPy writes array data in bounded chunks."""
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=target.name + ".", suffix=".tmp.npz", dir=target.parent
    )
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        np.savez_compressed(
            tmp,
            rgb=rgb_map,
            flow=flow_map,
            times=times,
            dt=dt,
            metadata=np.asarray(json.dumps(metadata, ensure_ascii=False)),
        )
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink()


def process_one(folder: Path, root: Path, output: Path,
                height: int, width: int, overwrite: bool) -> dict:
    import cv2

    video = folder / "gelsight.mp4"
    time_file = folder / "time2.npy"
    recording_id = folder.relative_to(root).as_posix()

    if not video.is_file() or not time_file.is_file():
        return {"id": recording_id, "status": "skip_missing_video_or_time"}

    target = output / recording_id.replace("/", "__") / "tactile_video_flow.npz"
    if target.is_file() and not overwrite:
        return {"id": recording_id, "status": "cached", "output": str(target)}

    times = load_times(time_file)
    frame_count = len(times)
    dt = np.diff(times).astype(np.float32)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return {"id": recording_id, "status": "error", "error": "Cannot open gelsight.mp4"}

    reported_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if reported_count > 0 and reported_count != frame_count:
        cap.release()
        return {
            "id": recording_id,
            "status": "skip_reported_count_mismatch",
            "reported_frames": reported_count,
            "timestamp_count": frame_count,
        }

    target.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = target.parent / ".memmap_tmp"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)

    rgb_path = temp_dir / "rgb_uint8.npy"
    flow_path = temp_dir / "flow_float16.npy"

    rgb_map = np.lib.format.open_memmap(
        rgb_path, mode="w+", dtype=np.uint8,
        shape=(frame_count, height, width, 3),
    )
    flow_map = np.lib.format.open_memmap(
        flow_path, mode="w+", dtype=np.float16,
        shape=(frame_count - 1, 2, height, width),
    )

    previous_gray = None
    decoded = 0
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(
                frame_rgb, (width, height), interpolation=cv2.INTER_AREA
            )
            gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)

            if decoded >= frame_count:
                raise ValueError("Decoded more frames than time2.npy timestamps")

            rgb_map[decoded] = resized

            if previous_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    previous_gray, gray, None,
                    0.5, 3, 15, 3, 5, 1.2, 0,
                )
                flow[..., 0] /= float(width)
                flow[..., 1] /= float(height)
                flow_map[decoded - 1] = np.moveaxis(flow, -1, 0).astype(
                    np.float16, copy=False
                )

            previous_gray = gray
            decoded += 1

            # Flush periodically so temporary data is durable during long jobs.
            if decoded % 256 == 0:
                rgb_map.flush()
                flow_map.flush()

        cap.release()

        if decoded != frame_count:
            return {
                "id": recording_id,
                "status": "skip_decoded_count_mismatch",
                "decoded_frames": decoded,
                "timestamp_count": frame_count,
            }

        rgb_map.flush()
        flow_map.flush()

        if not np.isfinite(flow_map).all():
            raise ValueError("Non-finite values found in float16 flow memmap")

        metadata = {
            "recording_id": recording_id,
            "source_video": str(video.resolve()),
            "source_time_file": str(time_file.resolve()),
            "video_size": video.stat().st_size,
            "video_mtime_ns": video.stat().st_mtime_ns,
            "time_size": time_file.stat().st_size,
            "time_mtime_ns": time_file.stat().st_mtime_ns,
            "video_sha256": sha256(video),
            "time_sha256": sha256(time_file),
            "frame_count": frame_count,
            "height": height,
            "width": width,
            "rgb_dtype": "uint8",
            "flow_dtype": "float16",
            "flow_layout": "T-1,2,H,W",
            "flow_units": "u/width,v/height per adjacent GelSight frame pair",
            "timestamps": "time2.npy shifted to zero",
            "labels": "not attached; annotation schema must be verified",
            "processing": "frame-wise decode + Farneback + disk memmap",
        }

        atomic_compressed_npz(target, rgb_map, flow_map, times, dt, metadata)

        return {
            "id": recording_id,
            "status": "written",
            "frames": frame_count,
            "flow_pairs": frame_count - 1,
            "rgb_dtype": "uint8",
            "flow_dtype": "float16",
            "output": str(target),
            "output_bytes": target.stat().st_size,
            "temp_memmap_bytes": rgb_path.stat().st_size + flow_path.stat().st_size,
        }
    finally:
        cap.release()
        del rgb_map
        del flow_map
        shutil.rmtree(temp_dir, ignore_errors=True)


def preprocess(root: Path, output: Path, height: int, width: int,
               limit: int | None, overwrite: bool) -> None:
    folders = find_recordings(root)
    if limit is not None:
        folders = folders[:limit]
    output.mkdir(parents=True, exist_ok=True)

    results = []
    counts: dict[str, int] = {}
    for index, folder in enumerate(folders, 1):
        try:
            result = process_one(folder, root, output, height, width, overwrite)
        except Exception as exc:
            result = {
                "id": folder.relative_to(root).as_posix(),
                "status": "error",
                "error": repr(exc),
            }
        counts[result["status"]] = counts.get(result["status"], 0) + 1
        results.append(result)
        print(f"[{index}/{len(folders)}] {json.dumps(result, ensure_ascii=False)}", flush=True)

    fields = sorted({key for row in results for key in row})
    manifest = output / "preprocess_manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    print("Summary:", json.dumps(counts, ensure_ascii=False))
    print("Manifest:", manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    audit_parser = sub.add_parser("audit")
    audit_parser.add_argument("--root", type=Path, required=True)
    audit_parser.add_argument("--report", type=Path)

    prep_parser = sub.add_parser("preprocess")
    prep_parser.add_argument("--root", type=Path, required=True)
    prep_parser.add_argument("--output", type=Path, required=True)
    prep_parser.add_argument("--height", type=int, default=128)
    prep_parser.add_argument("--width", type=int, default=96)
    prep_parser.add_argument("--limit", type=int)
    prep_parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()
    if not args.root.is_dir():
        parser.error(f"Dataset root does not exist: {args.root}")

    if args.command == "audit":
        audit(args.root, args.report)
    else:
        if args.height < 16 or args.width < 16:
            parser.error("height and width must be >=16")
        if args.limit is not None and args.limit < 1:
            parser.error("--limit must be positive")
        preprocess(args.root, args.output, args.height, args.width,
                    args.limit, args.overwrite)


if __name__ == "__main__":
    main()
