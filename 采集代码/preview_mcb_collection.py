"""Validate and preview an existing, format-separated GelSight collection.

Only reads rgb_png/depth_npz/flow_npz/signals_csv/metadata_json. New artifacts
are written under <run>/preview. Flow arrows use the saved pixel displacement
at 1x; the explicitly labelled color scale is fixed across episodes.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tactile_optical_flow import FINGERS, _load_rgb, _load_signals, _require_cv2


def _font(size=16):
    for path in (Path(r"C:\Windows\Fonts\arial.ttf"),
                 Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")):
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _vector(value, scale=1):
    return "[" + ", ".join(f"{float(v) * scale:.3f}" for v in value) + "]"


def load_episode(run: Path, metadata_path: Path, expected_frames=32):
    """Validate actual arrays and alignment; never rely on metadata alone."""
    relative = metadata_path.relative_to(run / "metadata_json")
    if len(relative.parts) != 3 or relative.name != "metadata.json":
        raise ValueError(f"Unexpected sample metadata path: {relative}")
    object_name, episode_name, _ = relative.parts
    logical = Path(object_name) / episode_name
    paths = {
        "metadata": metadata_path,
        "rgb": run / "rgb_png" / logical,
        "depth": run / "depth_npz" / logical / "tactile_depth.npz",
        "flow": run / "flow_npz" / logical / "tactile_flow.npz",
        "signals": run / "signals_csv" / logical / "signals.csv",
    }
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("collection_complete") is not True:
        raise ValueError("Metadata collection_complete is not true")
    if metadata.get("tactile_images_saved") is not True or metadata.get("optical_flow_saved") is not True:
        raise ValueError("Metadata must confirm tactile images and optical flow were saved")
    rgb = _load_rgb(paths["rgb"])
    shape = rgb[FINGERS[0]].shape
    frames, height, width, _ = shape
    if frames != expected_frames or metadata.get("num_frames") != frames:
        raise ValueError(f"Expected {expected_frames} RGB frames, got {frames}")
    times, contacts = _load_signals(paths["signals"], frames)
    if contacts is None:
        raise ValueError("Signals must include contact_count_f1/f2/f3")
    with paths["signals"].open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    contact_counts = np.asarray([[int(row[f"contact_count_f{i}"])
                                 for i in (1, 2, 3)] for row in rows])
    forces = np.asarray([[float(row[f"normal_force_f{i}"])
                          for i in (1, 2, 3)] for row in rows])
    if not np.isfinite(forces).all():
        raise ValueError("Signals contain a nonfinite normal force")
    with np.load(paths["depth"], allow_pickle=False) as archive:
        depth = {finger: archive[finger] for finger in FINGERS}
    with np.load(paths["flow"], allow_pickle=False) as archive:
        flow = {key: archive[key] for key in archive.files}
    pairs = frames - 1
    if str(flow["units"].item()) != "pixels/frame_pair":
        raise ValueError("Flow units must be pixels/frame_pair")
    flow_metadata = json.loads(str(flow["metadata"].item()))
    if flow_metadata.get("channels") != ["dx", "dy"]:
        raise ValueError("Flow channels must be ordered [dx, dy]")
    for key, expected in (("frame_from", np.arange(pairs)),
                          ("frame_to", np.arange(1, frames)),
                          ("contact_start", contacts[:-1]),
                          ("contact_end", contacts[1:]),
                          ("contact_pair", contacts[:-1] & contacts[1:])):
        if not np.array_equal(flow[key], expected):
            raise ValueError(f"Flow {key} does not align with signals/RGB")
    for key, expected in (("time_start", times[:-1]),
                          ("time_end", times[1:]), ("dt", np.diff(times))):
        if flow[key].shape != expected.shape or not np.allclose(
                flow[key], expected, rtol=0, atol=1e-6):
            raise ValueError(f"Flow {key} does not align with signals time")
    stats = {}
    quality_notes = []
    pair_maximum = np.zeros(pairs)
    for finger in FINGERS:
        if depth[finger].shape != (frames, height, width):
            raise ValueError(f"{finger}: incorrect depth shape {depth[finger].shape}")
        if depth[finger].dtype != np.float32 or not np.isfinite(depth[finger]).all():
            raise ValueError(f"{finger}: depth must be finite float32")
        values = flow[finger]
        if values.shape != (pairs, height, width, 2) or values.dtype != np.float32:
            raise ValueError(f"{finger}: flow must be float32 [T-1,H,W,2]")
        if not np.isfinite(values).all():
            raise ValueError(f"{finger}: nonfinite optical flow")
        magnitude = np.linalg.norm(values, axis=-1)
        pair_maximum = np.maximum(pair_maximum, magnitude.max(axis=(1, 2)))
        finger_index = FINGERS.index(finger)
        depth_nonzero_frames = np.any(np.abs(depth[finger]) > 1e-8, axis=(1, 2))
        rgb_delta = np.abs(rgb[finger].astype(np.int16) - rgb[finger][0].astype(np.int16))
        rgb_changed_pixels = np.any(rgb_delta > 0, axis=-1)
        stats[finger] = {
            "rgb_shape": list(rgb[finger].shape),
            "depth_shape": list(depth[finger].shape),
            "depth_dtype": str(depth[finger].dtype),
            "depth_max_normalized": float(depth[finger].max()),
            "depth_nonzero_frames": int(depth_nonzero_frames.sum()),
            "depth_nonzero_threshold": 1e-8,
            "rgb_changed_frames_from_first": int(rgb_changed_pixels.any(axis=(1, 2)).sum()),
            "rgb_changed_pixel_count_per_frame_from_first": rgb_changed_pixels.sum(axis=(1, 2)).tolist(),
            "rgb_max_abs_channel_change_from_first_uint8": int(rgb_delta.max()),
            "flow_shape": list(values.shape), "flow_dtype": str(values.dtype),
            "flow_magnitude_mean_px": float(magnitude.mean()),
            "flow_magnitude_p99_px": float(np.percentile(magnitude, 99)),
            "flow_magnitude_max_px": float(magnitude.max()),
            "frames_in_contact": int(contacts[:, finger_index].sum()),
            "contact_pairs": int(flow["contact_pair"][:, finger_index].sum()),
            "max_normal_force_N": float(forces[:, finger_index].max()),
        }
        if stats[finger]["frames_in_contact"] > 0 and not depth_nonzero_frames.any():
            quality_notes.append({
                "finger": finger,
                "code": "physics_contact_without_nonzero_height",
                "message": "Physics contact is recorded but every saved Taxim height map is zero; "
                           "contact constraints alone do not verify tactile deformation.",
            })
    pose = metadata.get("pose_config", {})
    report = {
        "object": object_name, "episode": episode_name,
        "structural_validation_passed": True,
        "collection_complete": metadata["collection_complete"],
        "valid_contact_sequence": metadata.get("valid") is True,
        "grasp_verified": metadata.get("grasp_verified", False) is True,
        "reason": metadata.get("reason"),
        "frames": frames, "frame_pairs": pairs, "signals_rows": len(rows),
        "time_seconds": [float(times[0]), float(times[-1])],
        "position_world_m": metadata.get("settled_object_position_m"),
        "initial_position_world_m": metadata.get("random_pose_command", {}).get("initial_position_m"),
        "position_center_world_m": pose.get("object_position_world_m"),
        "position_offset_world_m": pose.get("object_position_offset_world_m"),
        "final_position_world_m": metadata.get("final_object_position_m"),
        "object_rpy_deg": pose.get("object_rpy_deg"),
        "contact_offset_object_m": pose.get("grasp_offset_object_m"),
        "flow_units": "pixels/frame_pair", "flow_channels": ["dx", "dy"],
        "flow_ground_truth": False,
        "depth_units": metadata.get("tactile_depth_units", "dimensionless"),
        "paths": {key: value.relative_to(run).as_posix() for key, value in paths.items()},
        "fingers": stats,
        "quality_notes": quality_notes,
    }
    return {
        "metadata": metadata, "report": report, "rgb": rgb, "flow": flow,
        "times": times, "contacts": contacts, "contact_counts": contact_counts,
        "forces": forces, "rows": rows,
        "representative_pair": int(pair_maximum.argmax()),
    }


def flow_image(flow, color_max, cv2):
    """No arrow gain: vector endpoints remain in original image coordinates."""
    dx, dy = flow[..., 0], flow[..., 1]
    magnitude = np.hypot(dx, dy)
    angle = np.mod(np.arctan2(dy, dx), 2 * np.pi)
    hsv = np.zeros((*dx.shape, 3), dtype=np.uint8)
    hsv[..., 0] = np.floor(angle * 180 / (2 * np.pi)).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(magnitude / color_max * 255, 0, 255).astype(np.uint8)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    # Subpixel line rendering keeps arrow endpoints honest even for small flow.
    for y in range(16, flow.shape[0], 32):
        for x in range(16, flow.shape[1], 32):
            endpoint = (int(round((x + dx[y, x]) * 256)),
                        int(round((y + dy[y, x]) * 256)))
            if magnitude[y, x] >= 0.5:
                cv2.arrowedLine(rgb, (x * 256, y * 256), endpoint,
                                (230, 230, 230), 1, cv2.LINE_AA, 8, 0.25)
    return Image.fromarray(rgb)


def render_frame(sample, frame, color_max, cv2, playback_fps=10):
    canvas = Image.new("RGB", (800, 1040), (20, 26, 35))
    draw = ImageDraw.Draw(canvas)
    font, small, title = _font(16), _font(13), _font(20)
    report = sample["report"]
    pair = min(frame, report["frame_pairs"] - 1)
    pose = sample["metadata"].get("pose_config", {})
    white, gray = (235, 240, 248), (173, 188, 205)
    draw.text((20, 12), f"GelSight capture | {report['episode']}", font=title, fill=white)
    name = report["object"]
    draw.text((20, 42), f"Object: {name[:90]}", font=small, fill=gray)
    draw.text((20, 64), "Settled position [m]: " + _vector(report.get("position_world_m") or []),
              font=font, fill=white)
    draw.text((20, 88), "RPY [deg]: " + _vector(pose.get("object_rpy_deg", [])) +
              "   Contact offset [mm]: " + _vector(pose.get("grasp_offset_object_m", []), 1000),
              font=font, fill=white)
    time = sample["times"][frame] - sample["times"][0]
    draw.text((20, 113), f"Frame {frame:02d}/{report['frames'] - 1} | t={time:.3f}s | "
              f"{sample['rows'][frame]['phase']}",
              font=font, fill=(122, 218, 175))
    draw.text((440, 116), f"Valid contact seq: {report['valid_contact_sequence']} | "
              f"Grasp verified: {report['grasp_verified']}", font=small, fill=(122, 218, 175))
    for index, finger in enumerate(FINGERS):
        x = 24 + 256 * index
        draw.text((x, 146), f"{finger} RGB | contacts={sample['contact_counts'][frame, index]}",
                  font=small, fill=white)
        image = Image.fromarray(sample["rgb"][finger][frame]).resize((240, 320))
        canvas.paste(image, (x, 168))
        draw.text((x, 503), f"Flow pair {pair:02d} -> {pair + 1:02d}", font=font, fill=white)
        image = flow_image(sample["flow"][finger][pair], color_max, cv2).resize((240, 320))
        canvas.paste(image, (x, 530))
        maximum = np.linalg.norm(sample["flow"][finger][pair], axis=-1).max()
        draw.text((x, 859), f"Max |flow|: {maximum:.5f} px/pair", font=small, fill=gray)
    draw.text((24, 883), f"Color: brightness 0..{color_max:g} px/pair; hue=direction (right=red, down=yellow-green).",
              font=small, fill=gray)
    draw.text((24, 904), "Arrows: actual 1x pixel displacement; shown only above 0.5 px. Small flow stays small.",
              font=small, fill=gray)
    zero_height_fingers = [note["finger"] for note in report["quality_notes"]
                           if note["code"] == "physics_contact_without_nonzero_height"]
    if zero_height_fingers:
        note = "Physics contact but zero height: " + ", ".join(zero_height_fingers) + ". Flow is RGB-estimated."
        draw.text((24, 925), note, font=small, fill=(255, 188, 112))
    else:
        draw.text((24, 925), "Estimated flow from synthetic RGB; normalized depth is dimensionless.",
                  font=small, fill=gray)
    chart = (180, 955, 774, 1006)
    draw.text((24, 959), "Normal force [N]", font=small, fill=gray)
    draw.rectangle(chart, outline=(65, 77, 93))
    maximum = max(float(sample["forces"].max()), 0.001)
    colors = ((255, 133, 122), (125, 219, 169), (124, 172, 255))
    for index in range(3):
        points = [(chart[0] + k / (report["frames"] - 1) * (chart[2] - chart[0]),
                   chart[3] - value / maximum * (chart[3] - chart[1] - 4))
                  for k, value in enumerate(sample["forces"][:, index])]
        draw.line(points, fill=colors[index], width=2)
    cursor = chart[0] + frame / (report["frames"] - 1) * (chart[2] - chart[0])
    draw.line([(cursor, chart[1]), (cursor, chart[3])], fill=white)
    draw.text((24, 1018), f"Force chart max={maximum:.3f} N | video playback={playback_fps:g} frames/s",
              font=small, fill=gray)
    return canvas


def create_preview(run, expected_frames=32, color_max=1.0, fps=10.0):
    run = Path(run).resolve()
    if not (run / "rgb_png").is_dir() or not (run / "metadata_json").is_dir():
        raise ValueError("Expected a by_type collection run with rgb_png and metadata_json")
    preview = (run / "preview").resolve()
    if preview.parent != run or preview.is_symlink():
        raise ValueError("Preview directory must stay directly inside the run")
    preview.mkdir(exist_ok=True)
    cv2 = _require_cv2()
    report = {
        "run": str(run), "layout": "by_type", "expected_frames_per_finger": expected_frames,
        "flow_visualization": {
            "color_max_pixels_per_pair": color_max, "arrow_gain": 1,
            "arrow_minimum_pixels": 0.5, "playback_fps": fps,
            "description": "Fixed flow color scale; 1x arrows; no numerical flow amplification",
        },
        "samples": [], "errors": [],
    }
    samples = []
    for metadata in sorted((run / "metadata_json").glob("*/episode_*/metadata.json")):
        try:
            sample = load_episode(run, metadata, expected_frames)
            samples.append(sample)
            report["samples"].append(sample["report"])
        except Exception as exc:
            report["errors"].append({"metadata": metadata.relative_to(run).as_posix(),
                                     "error": f"{type(exc).__name__}: {exc}"})
    report["sample_count"] = len(samples)
    report["all_structurally_valid"] = bool(samples) and not report["errors"]
    report["all_contact_sequences_valid"] = bool(samples) and all(
        s["report"]["valid_contact_sequence"] for s in samples)
    report["all_grasps_verified"] = bool(samples) and all(
        s["report"]["grasp_verified"] for s in samples)
    report["quality_notes"] = [
        {"object": sample["report"]["object"], "episode": sample["report"]["episode"], **note}
        for sample in samples for note in sample["report"]["quality_notes"]
    ]
    report["artifacts"] = []
    if samples:
        # Keep the overview readable: at most two columns, one full sample per tile.
        columns = min(2, len(samples))
        rows = (len(samples) + columns - 1) // columns
        overview = Image.new("RGB", (800 * columns, 1040 * rows), (20, 26, 35))
        for index, sample in enumerate(samples):
            tile = render_frame(sample, sample["representative_pair"], color_max, cv2, fps)
            name = f"{sample['report']['object']}_{sample['report']['episode']}.png"
            tile.save(preview / name)
            overview.paste(tile, ((index % columns) * 800, (index // columns) * 1040))
        overview.save(preview / "overview.png")
        report["artifacts"].append("preview/overview.png")
        # OpenCV's Windows VideoWriter may reject Unicode paths. Encode to an
        # ASCII temporary location, then copy the complete file to this run.
        with tempfile.TemporaryDirectory(prefix="mcb_tactile_preview_") as temporary:
            temporary_video = Path(temporary) / "tactile_flow.mp4"
            writer = cv2.VideoWriter(str(temporary_video), cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps, (800, 1040))
            if not writer.isOpened():
                raise RuntimeError("OpenCV could not open the MP4 encoder")
            written = 0
            try:
                for sample in samples:
                    for frame in range(sample["report"]["frames"]):
                        rgb = np.asarray(render_frame(sample, frame, color_max, cv2, fps))
                        writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                        written += 1
            finally:
                writer.release()
            capture = cv2.VideoCapture(str(temporary_video))
            actual_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            ok, first_frame = capture.read()
            capture.release()
            if actual_frames != written or not ok or first_frame.shape[:2] != (1040, 800):
                raise RuntimeError(f"MP4 verification failed: {actual_frames}/{written} frames")
            shutil.copyfile(temporary_video, preview / "tactile_flow.mp4")
        report["video_frames"] = written
        report["video_duration_seconds"] = written / fps
        report["artifacts"].append("preview/tactile_flow.mp4")
    (preview / "validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--expected-frames", type=int, default=32)
    parser.add_argument("--flow-color-max", type=float, default=1.0,
                        help="Fixed color brightness scale in pixels/frame_pair")
    parser.add_argument("--fps", type=float, default=10.0,
                        help="Preview playback fps; data timestamps are unchanged")
    args = parser.parse_args(argv)
    if args.expected_frames < 2 or args.flow_color_max <= 0 or args.fps <= 0:
        parser.error("Expected frames >= 2, color max > 0, and fps > 0")
    report = create_preview(args.run, args.expected_frames, args.flow_color_max, args.fps)
    print(json.dumps({key: report[key] for key in (
        "sample_count", "all_structurally_valid", "all_contact_sequences_valid",
        "all_grasps_verified", "quality_notes", "errors", "artifacts")},
        ensure_ascii=False, indent=2))
    return 0 if report["all_structurally_valid"] and report["all_contact_sequences_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
