#!/usr/bin/env python
"""Collect the YCB three-finger tactile plan with independent CPU workers.

Run build_ycb_tactile_catalog.py first. This script creates per-object robot
MJCF scenes, checks them with MuJoCo, estimates storage, then runs the existing
collector as isolated worker processes. Each worker owns its model, data, and
Taxim renderers; episode indices are sharded without overlap.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import importlib.util
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import sys
import zlib

import numpy as np


PROJECT = Path(r"E:\project\p1\deepgelsight")
DEFAULT_OBJECT_ROOT = PROJECT / "object" / "ycb"
DEFAULT_TEMPLATE = PROJECT / "xml" / "ur3_with_threefinger_gelsight_table_tomato_underactuated.xml"
DEFAULT_CATALOG = PROJECT / "dataset_underactuated" / "plans" / "ycb_tactile_catalog" / "ycb_tactile_catalog.json"
DEFAULT_SCENES = PROJECT / "xml" / "generated_ycb"
DEFAULT_COLLECTOR = PROJECT / "collect_underactuated_tactile_dataset_pose_sweep_v3.py"
DEFAULT_OUTPUT_ROOT = PROJECT / "dataset_underactuated" / "runs"
DEFAULT_WORKERS = 2
DEFAULT_MAX_CENTERS = 24
DEFAULT_SPACING_M = 0.005
DEFAULT_RADIAL_CAP_M = 0.015
DEFAULT_AXIAL_CAP_M = 0.030
DEFAULT_REPEATS = 3
DEFAULT_ROLLS = (0.0, 30.0, 60.0, 90.0)
DEFAULT_EPISODE_MB = 20.0
DEFAULT_OBJECT_COUNT = 10
# Keep the default set shape-diverse; --objects overrides it exactly.
DEFAULT_OBJECTS = (
    "apple", "banana", "cracker_box", "master_chef_can", "mug",
    "mustard_bottle", "power_drill", "sponge", "sugar_box", "tennis_ball",
)


def load_python_file(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import helper module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def sample_center_offsets(extents: np.ndarray, spacing: float, radial_cap: float,
                          axial_cap: float, max_centers: int) -> list[list[float]]:
    radial = min(radial_cap, 0.44 * float(min(extents[0], extents[1])))
    axial = min(axial_cap, 0.30 * float(extents[2]))
    xs = np.arange(-radial, radial + spacing / 2, spacing)
    zs = np.arange(-axial, axial + spacing / 2, spacing)
    candidates = np.asarray([
        (x, y, z)
        for x in xs for y in xs
        for z in zs
        if x * x + y * y <= radial * radial + 1e-12
    ], dtype=float)
    if not max_centers or len(candidates) <= max_centers:
        return candidates.round(5).tolist()

    # Deterministic farthest-point sampling spreads the cap over x/y/z instead
    # of taking only the center-out prefix of the dense grid.
    scale = np.maximum(extents * 0.5, spacing)
    normalized = candidates / scale
    first = int(np.argmin(np.linalg.norm(normalized, axis=1)))
    selected = [first]
    min_dist_sq = np.sum((normalized - normalized[first]) ** 2, axis=1)
    while len(selected) < max_centers:
        next_idx = int(np.argmax(min_dist_sq))
        selected.append(next_idx)
        dist_sq = np.sum((normalized - normalized[next_idx]) ** 2, axis=1)
        min_dist_sq = np.minimum(min_dist_sq, dist_sq)
    return candidates[selected].round(5).tolist()


def make_jobs(catalog: dict, selected: list[str], args, scene_builder) -> list[dict]:
    rows = {row["object"]: row for row in catalog["objects"] if row.get("valid")}
    missing = [name for name in selected if name not in rows]
    if missing:
        raise ValueError(f"Objects not found or invalid in catalog: {missing}")

    jobs = []
    for name in selected:
        row = rows[name]
        if row.get("compile_error"):
            raise RuntimeError(f"MuJoCo standalone compile failed for {name}: {row['compile_error']}")
        scene = scene_builder.build_scene(
            args.template, args.object_root, name,
            args.scenes_dir / f"ur3_threefinger_{name}.xml",
        )
        import mujoco

        mujoco.MjModel.from_xml_path(str(scene))  # fail before data collection starts

        poses = row.get("stable_poses", [])
        if args.pose_limit > 0:
            poses = poses[:args.pose_limit]
        if not poses:
            raise ValueError(f"No pose configs in catalog for {name}")
        centers = sample_center_offsets(
            np.asarray(row["collision_extents_m"], dtype=float),
            args.spacing, args.radial_cap, args.axial_cap,
            args.max_centers_per_pose,
        )
        configs = []
        for center in centers:
            for roll in args.rolls:
                configs.append({
                    "grasp_offset_object_m": center,
                    "approach_azimuth_deg": 0.0,
                    "approach_tilt_deg": 0.0,
                    "gripper_roll_deg": float(roll),
                })
        job_poses = []
        for pose in poses:
            if "initial_body_z_m" not in pose:
                raise ValueError(f"Catalog pose {name}/{pose.get('pose_id')} lacks initial_body_z_m")
            job_poses.append({
                "pose_id": pose["pose_id"],
                "quat_wxyz": pose["quat_wxyz"],
                "initial_body_z_m": float(pose["initial_body_z_m"]),
            })
        episodes = len(configs) * len(job_poses) * args.repeats
        jobs.append({
            "object": name,
            "scene": str(scene),
            "center_local_m": row["center_local_m"],
            "extents_m": row["collision_extents_m"],
            "centers": centers,
            "poses": job_poses,
            "configs": configs,
            "repeats": args.repeats,
            "episodes": episodes,
        })
    return jobs


def worker_run(task: dict) -> tuple[int, int]:
    worker_id = int(task["worker_id"])
    worker_count = int(task["worker_count"])
    root = Path(task["output_root"])
    collector_path = Path(task["collector"])
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["NUMBA_NUM_THREADS"] = "1"
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    collector = load_python_file(collector_path, f"tactile_collector_worker_{worker_id}")
    collector.RUN_OUTPUT_ROOT = root
    collector.SHOW_VIEWER = False
    collector.EPISODE_SHARD_INDEX = worker_id
    collector.EPISODE_SHARD_COUNT = worker_count
    collector.RESUME_COMPLETED_EPISODES = bool(task["resume"])
    collector.REPEATS_PER_CONFIG = int(task["repeats"])
    collector.GRIPPER_ROLLS_DEG = list(task["rolls"])
    collector.CONTACT_GRID_SPACING_M = float(task["spacing"])
    collector.BASE_SEED = int(task["base_seed"])

    attempted = 0
    for job in task["jobs"]:
        print(f"\n[WORKER {worker_id + 1}/{worker_count}] OBJECT {job['object']}", flush=True)
        collector.XML_PATH = Path(job["scene"])
        collector.DATASET_OBJECT_LABEL = job["object"]
        collector.GRASP_CENTER_OBJECT = np.asarray(job["center_local_m"], dtype=float)
        collector.OBJECT_POSES = {
            pose["pose_id"]: list(pose["quat_wxyz"]) for pose in job["poses"]
        }
        collector.OBJECT_POSE_DROP_Z = {
            pose["pose_id"]: float(pose["initial_body_z_m"]) for pose in job["poses"]
        }
        collector.POSE_STAGES = [{
            "name": "dense_contact_grid",
            "configs": job["configs"],
        }]
        collector.NUM_EPISODES = (
            len(job["configs"]) * len(job["poses"]) * int(job["repeats"])
        )
        object_seed = int(task["base_seed"]) + zlib.crc32(job["object"].encode("utf-8"))
        collector.BASE_SEED = object_seed
        collector.TACTILE_RENDERERS.clear()
        collector.main()
        attempted += job["episodes"]
    return worker_id, attempted


def merge_summaries(run_root: Path) -> Path:
    inputs = sorted(run_root.glob("collection_summary_*_worker_*.csv"))
    combined = run_root / "collection_summary.csv"
    if not inputs:
        return combined
    with combined.open("w", newline="", encoding="utf-8-sig") as out_handle:
        writer = None
        for path in inputs:
            with path.open("r", newline="", encoding="utf-8-sig") as in_handle:
                reader = csv.DictReader(in_handle)
                if writer is None:
                    writer = csv.DictWriter(out_handle, fieldnames=reader.fieldnames)
                    writer.writeheader()
                writer.writerows(reader)
    return combined


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--object-root", type=Path, default=DEFAULT_OBJECT_ROOT)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--scenes-dir", type=Path, default=DEFAULT_SCENES)
    parser.add_argument("--collector", type=Path, default=DEFAULT_COLLECTOR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--objects", nargs="*", default=None,
                        help="YCB object names; overrides the 10-object default")
    parser.add_argument("--object-count", type=int, default=DEFAULT_OBJECT_COUNT,
                        help="Number of default objects; 0 means all valid catalog objects")
    parser.add_argument("--all-objects", action="store_true",
                        help="Collect every valid catalog object (same as --object-count 0)")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--max-centers-per-pose", type=int, default=DEFAULT_MAX_CENTERS,
                        help="0 keeps the full 5 mm grid; default caps each pose at 24 spread-out centers")
    parser.add_argument("--pose-limit", type=int, default=0,
                        help="0 keeps all catalog poses (identity plus stable poses)")
    parser.add_argument("--spacing", type=float, default=DEFAULT_SPACING_M)
    parser.add_argument("--radial-cap", type=float, default=DEFAULT_RADIAL_CAP_M)
    parser.add_argument("--axial-cap", type=float, default=DEFAULT_AXIAL_CAP_M)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--rolls", type=float, nargs="+", default=DEFAULT_ROLLS)
    parser.add_argument("--episode-mb", type=float, default=DEFAULT_EPISODE_MB)
    parser.add_argument("--base-seed", type=int, default=20260927)
    parser.add_argument("--run-id", default=None,
                        help="Timestamp-like directory name; specify the same id to resume")
    parser.add_argument("--resume", action="store_true",
                        help="Skip episodes with collection_complete metadata under an existing run-id")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build/compile scenes and print the job/storage plan without simulating")
    parser.add_argument("--allow-over-budget", action="store_true",
                        help="Run even if the estimate exceeds available disk space")
    args = parser.parse_args()

    for path, label in ((args.catalog, "catalog"), (args.template, "template"),
                        (args.collector, "collector"), (args.object_root, "object root")):
        if not path.exists():
            parser.error(f"Missing {label}: {path}")
    if args.workers < 1 or args.repeats < 1:
        parser.error("workers and repeats must be at least 1")
    if args.resume and not args.run_id:
        parser.error("--resume requires an explicit --run-id")

    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    valid_names = [row["object"] for row in catalog["objects"] if row.get("valid")]
    if args.objects:
        selected = args.objects
    elif args.all_objects or args.object_count == 0:
        selected = valid_names
    else:
        if args.object_count < 1:
            parser.error("--object-count must be positive, or 0 for all objects")
        preferred = [name for name in DEFAULT_OBJECTS if name in valid_names]
        selected = (preferred + [name for name in valid_names if name not in preferred])[:args.object_count]
    if not selected:
        parser.error("No valid YCB objects selected")
    unknown = sorted(set(selected) - set(valid_names))
    if unknown:
        parser.error(f"Objects absent/invalid in catalog: {unknown}")

    if args.run_id:
        run_id = args.run_id
    else:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_ycb"
    run_root = args.output_root / run_id
    if run_root.exists() and not args.resume:
        parser.error(f"Run directory already exists; choose a new --run-id or use --resume: {run_root}")
    if args.resume and not run_root.is_dir():
        parser.error(f"Cannot resume; run directory does not exist: {run_root}")

    scene_builder = load_python_file(Path(__file__).with_name("build_ycb_robot_scenes.py"), "ycb_scene_builder_runtime")
    jobs = make_jobs(catalog, selected, args, scene_builder)
    total_episodes = sum(job["episodes"] for job in jobs)
    estimated_bytes = int(total_episodes * args.episode_mb * 1_000_000)
    disk_anchor = args.output_root
    while not disk_anchor.exists() and disk_anchor.parent != disk_anchor:
        disk_anchor = disk_anchor.parent
    free_bytes = shutil.disk_usage(disk_anchor).free
    print(f"Objects: {len(jobs)}; workers: {args.workers}")
    print("Selected objects: " + ", ".join(selected))
    print(f"Planned episodes: {total_episodes:,}")
    print(f"Estimated output: {estimated_bytes / 1e9:,.1f} GB at {args.episode_mb:g} MB/episode")
    print(f"Available disk: {free_bytes / 1e9:,.1f} GB")
    if estimated_bytes > free_bytes * 0.9 and not args.allow_over_budget and not args.dry_run:
        raise SystemExit(
            "Disk budget check stopped collection before launch. Select fewer objects/centers/poses, "
            "or explicitly pass --allow-over-budget after arranging more storage."
        )

    run_root.mkdir(parents=True, exist_ok=True)
    run_config = {
        "run_id": run_id,
        "objects": selected,
        "workers": args.workers,
        "repeats": args.repeats,
        "rolls_deg": list(args.rolls),
        "max_centers_per_pose": args.max_centers_per_pose or None,
        "pose_limit": args.pose_limit or None,
        "spacing_m": args.spacing,
        "radial_cap_m": args.radial_cap,
        "axial_cap_m": args.axial_cap,
        "episode_count": total_episodes,
        "episode_size_assumption_mb": args.episode_mb,
        "estimated_storage_bytes": estimated_bytes,
        "base_seed": args.base_seed,
    }
    run_config_path = run_root / "run_config.json"
    if args.resume:
        if not run_config_path.is_file():
            parser.error(f"Resume metadata is missing: {run_config_path}")
        prior_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        comparable_keys = (
            "objects", "workers", "repeats", "rolls_deg", "max_centers_per_pose",
            "pose_limit", "spacing_m", "radial_cap_m", "axial_cap_m",
            "episode_count", "base_seed",
        )
        mismatches = [key for key in comparable_keys if prior_config.get(key) != run_config.get(key)]
        if mismatches:
            parser.error(
                "Resume settings differ from the original run for: " + ", ".join(mismatches)
            )
    else:
        run_config_path.write_text(
            json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    (run_root / "collection_jobs.json").write_text(
        json.dumps(jobs, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if args.dry_run:
        print("Dry run only: models compiled and job manifest written; no episodes were started.")
        print(f"Job manifest: {run_root / 'collection_jobs.json'}")
        return 0

    tasks = []
    for worker_id in range(args.workers):
        tasks.append({
            "worker_id": worker_id,
            "worker_count": args.workers,
            "jobs": jobs,
            "output_root": str(run_root),
            "collector": str(args.collector),
            "resume": args.resume,
            "repeats": args.repeats,
            "rolls": list(args.rolls),
            "spacing": args.spacing,
            "base_seed": args.base_seed,
        })
    errors = []
    with ProcessPoolExecutor(max_workers=args.workers,
                             mp_context=mp.get_context("spawn")) as executor:
        futures = [executor.submit(worker_run, task) for task in tasks]
        for future in as_completed(futures):
            try:
                worker_id, attempted = future.result()
                print(f"Worker {worker_id + 1} finished; assigned episode attempts: {attempted:,}")
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
                print(f"Worker failed: {type(exc).__name__}: {exc}")
    summary_path = merge_summaries(run_root)
    (run_root / "launcher_status.json").write_text(json.dumps({
        "complete": not errors,
        "errors": errors,
        "summary_csv": str(summary_path),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Run folder: {run_root}")
    print(f"Merged summary: {summary_path}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
