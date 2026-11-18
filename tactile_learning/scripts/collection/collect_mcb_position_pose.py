#!/usr/bin/env python
"""Collect MCB position × orientation × contact sweeps with three GelSight streams.

Without --run this prints the sampling plan. Positions are WORLD offsets from
the collection center (--center-m); contact offsets are in OBJECT coordinates. Object
rotations use Rz(yaw) @ Ry(pitch) @ Rx(roll), quaternion order w,x,y,z.

The default by_type layout puts PNG frames in rgb_png/, numeric depth/flow
archives in depth_npz/ and flow_npz/, CSV in signals_csv/, and JSON in
metadata_json/. Each data folder uses the same object/episode identifiers.
Use --all-categories to select one representative OBJ from every raw category,
including categories without a pre-generated XML or numbered launcher.
"""
from __future__ import annotations

import argparse
import csv
import copy
from datetime import datetime
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import tempfile

import numpy as np

import collect_mcb_any_object as models
import collect_mcb_gelsight_dataset as adapter

HERE = Path(__file__).resolve().parent
# Scripts can live directly beside MCB_A/ on a server, or in a code subfolder.
DEFAULT_OUTPUT = (HERE if (HERE / "MCB_A").is_dir() else HERE.parent) / "mcb_position_pose_runs"


def cross_points(distance: float) -> list[list[float]]:
    """A center and positive/negative offsets on each axis (not a cube grid)."""
    result = [[0.0, 0.0, 0.0]]
    for axis in range(3):
        for sign in (1.0, -1.0):
            point = [0.0, 0.0, 0.0]
            point[axis] = sign * distance
            result.append(point)
    return result


def rpy_quaternion(degrees: list[float]) -> list[float]:
    roll, pitch, yaw = np.deg2rad(np.asarray(degrees, dtype=float)) / 2
    cr, cp, cy = math.cos(roll), math.cos(pitch), math.cos(yaw)
    sr, sp, sy = math.sin(roll), math.sin(pitch), math.sin(yaw)
    q = np.array([cr*cp*cy + sr*sp*sy, sr*cp*cy - cr*sp*sy,
                  cr*sp*cy + sr*cp*sy, cr*cp*sy - sr*sp*cy])
    return (q / np.linalg.norm(q)).tolist()


def finite_points(points, name: str) -> list[list[float]]:
    result = []
    for point in points:
        array = np.asarray(point, dtype=float)
        if array.shape != (3,) or not np.isfinite(array).all():
            raise ValueError(f"{name} requires three finite numbers")
        converted = array.tolist()
        if converted not in result:
            result.append(converted)
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result


def make_configs(args) -> tuple[list[dict], dict]:
    world_mm = finite_points(args.position_mm or cross_points(10.0), "--position-mm")
    contact_mm = finite_points(args.contact_mm or cross_points(5.0), "--contact-mm")
    rpy_deg = finite_points(args.rpy_deg or cross_points(90.0), "--rpy-deg")
    rolls = list(dict.fromkeys(float(value) for value in args.rolls_deg))
    if not rolls or not np.isfinite(rolls).all():
        raise ValueError("--rolls-deg requires finite angles")
    configs = []
    center = None if args.center_m is None else finite_points([args.center_m], "--center-m")[0]
    for p, r, c, roll in itertools.product(
        range(len(world_mm)), range(len(rpy_deg)), range(len(contact_mm)), rolls
    ):
        configs.append({
            "position_group": f"position_{p:03d}",
            "pose_group": f"orientation_{r:03d}",
            "contact_group": f"contact_{c:03d}",
            "object_position_offset_world_m": (np.asarray(world_mm[p]) / 1000).tolist(),
            "object_rpy_deg": rpy_deg[r],
            "object_quat_wxyz": rpy_quaternion(rpy_deg[r]),
            "grasp_offset_object_m": (np.asarray(contact_mm[c]) / 1000).tolist(),
            "approach_azimuth_deg": 0.0,
            "approach_tilt_deg": 0.0,
            "gripper_roll_deg": roll,
            "follow_object_orientation": bool(args.follow_object_orientation),
        })
        if center is not None:
            configs[-1]["object_position_world_m"] = center
    full_count = len(configs)
    if args.max_configs is not None and args.max_configs < full_count:
        # Evenly spaced indices include both endpoints. A small pilot then
        # changes world position, object orientation AND contact location.
        indices = np.linspace(0, full_count - 1, args.max_configs, dtype=int)
        configs = [configs[int(i)] for i in indices]
    dimensions = {
        "world_position_offsets_mm": world_mm,
        "object_rpy_deg": rpy_deg,
        "contact_offsets_object_mm": contact_mm,
        "gripper_rolls_deg": rolls,
        "full_combination_count": full_count,
        "selected_combination_count": len(configs),
        "repeats": args.repeats,
        "episodes": len(configs) * args.repeats,
    }
    return configs, dimensions


def model_fingerprint(xml_path: Path) -> dict:
    import xml.etree.ElementTree as ET
    sources = [xml_path]
    root = ET.parse(xml_path).getroot()
    compiler = root.find("compiler")
    mesh_dir = Path(compiler.get("meshdir", ".") if compiler is not None else ".")
    if not mesh_dir.is_absolute():
        mesh_dir = xml_path.parent / mesh_dir
    for mesh in root.findall(".//mesh"):
        file_name = mesh.get("file")
        if file_name:
            mesh_path = Path(file_name)
            sources.append(mesh_path if mesh_path.is_absolute() else mesh_dir / mesh_path)
    return {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(set(sources))}


def save_plan(output: Path, plan: dict, resume: bool) -> None:
    by_type = plan.get("output_layout") == "by_type"
    path = output / "metadata_json" / "collection_plan.json" if by_type else output / "collection_plan.json"
    csv_path = output / "signals_csv" / "collection_plan.csv" if by_type else output / "collection_plan.csv"
    if output.exists() and any(output.iterdir()):
        if not resume or not path.is_file():
            raise ValueError(f"输出目录已有文件，请换目录或使用 --resume：{output}")
        prior = json.loads(path.read_text(encoding="utf-8"))
        if prior != plan:
            raise ValueError("断点续采参数或模型发生变化，请使用新的输出目录")
        return
    if resume:
        raise ValueError("--resume 需要已有 collection_plan.json 的原输出目录")
    output.mkdir(parents=True, exist_ok=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    write_plan_csv(csv_path, plan)


def write_plan_csv(csv_path: Path, plan: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["episode", "repeat", "position_group", "pose_group", "contact_group",
                         "world_offset_m", "quaternion_wxyz", "contact_offset_object_m", "roll_deg"])
        for index, cfg in enumerate(plan["configs"]):
            for repeat in range(plan["dimensions"]["repeats"]):
                writer.writerow([index * plan["dimensions"]["repeats"] + repeat, repeat,
                                 cfg["position_group"], cfg["pose_group"], cfg["contact_group"],
                                 json.dumps(cfg["object_position_offset_world_m"]),
                                 json.dumps(cfg["object_quat_wxyz"]),
                                 json.dumps(cfg["grasp_offset_object_m"]), cfg["gripper_roll_deg"]])


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--obj", help="单个 OBJ 路径；不填且未选择批量模式时打开选择器")
    selection.add_argument("--all-categories", action="store_true", help="遍历原始模型库所有类别，每类采集一个代表物体")
    parser.add_argument("--dataset-root", type=Path, default=models.DATASET_ROOT, help="MCB 原始 train/test 模型根目录")
    parser.add_argument("--start-category", type=int, default=1, help="批量模式起始类别编号（1-based）")
    parser.add_argument("--limit-categories", type=int, help="批量模式最多采集多少类；省略时全部")
    parser.add_argument("--output", type=Path, help="新的输出目录；续采时指定原目录")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--run", action="store_true", help="运行采集；默认只显示计划")
    actions.add_argument("--check-only", action="store_true", help="只检查模型和采样配置")
    parser.add_argument("--headless", action="store_true", help="采集时隐藏 MuJoCo viewer")
    parser.add_argument("--layout", choices=("by_type", "by_episode"), default="by_type",
                        help="新数据按类型分目录（默认）或按样本内分格式子目录")
    parser.add_argument("--center-m", type=float, nargs=3, metavar=("X", "Y", "Z"),
                        help="采集基准世界中心，单位米；默认模型XY和Z=0.84m")
    parser.add_argument("--position-mm", type=float, nargs=3, action="append", metavar=("X", "Y", "Z"),
                        help="物体世界位置偏移毫米，可重复；默认中心和各轴±10mm")
    parser.add_argument("--rpy-deg", type=float, nargs=3, action="append", metavar=("RX", "RY", "RZ"),
                        help="物体绕世界 XYZ 的欧拉角，可重复；默认零和各轴±90°")
    parser.add_argument("--contact-mm", type=float, nargs=3, action="append", metavar=("X", "Y", "Z"),
                        help="物体系中的抓取中心偏移毫米，可重复；默认中心和各轴±5mm")
    parser.add_argument("--rolls-deg", type=float, nargs="+", default=[0.0])
    parser.add_argument("--follow-object-orientation", action="store_true",
                        help="夹爪跟随物体旋转；默认保持基准朝向以改变相对接触方向")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-configs", type=int, help="均匀选取少量组合用于试采")
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--resume", action="store_true", help="相同计划下跳过已完成 episode")
    parser.add_argument("--free-floating", action="store_true", help="接触时允许物体自由运动；默认虚拟固定")
    parser.add_argument("--sensor-clearance-mm", type=float, default=1.0,
                        help="不同手指传感器组件的最小间距，默认1mm；接近时拒绝整段样本")
    args = parser.parse_args(argv)
    if args.repeats < 1 or (args.max_configs is not None and args.max_configs < 1):
        parser.error("--repeats 和 --max-configs 必须为正整数")
    if args.resume and (not args.run or args.output is None):
        parser.error("--resume 必须同时指定 --run 和原 --output")
    if args.start_category < 1 or (args.limit_categories is not None and args.limit_categories < 1):
        parser.error("类别起点和数量必须为正整数")
    if not args.all_categories and (args.start_category != 1 or args.limit_categories is not None):
        parser.error("--start-category/--limit-categories 必须搭配 --all-categories")
    if not math.isfinite(args.sensor_clearance_mm) or args.sensor_clearance_mm <= 0:
        parser.error("--sensor-clearance-mm 必须是大于0的有限数值")
    return args


def prepare_object(args, obj_path, label=None, configs=None, dimensions=None,
                   category=None, category_index=None) -> dict:
    """Compile one raw OBJ and build its complete, reproducible sampling plan."""
    if configs is None or dimensions is None:
        configs, dimensions = make_configs(args)
    configs, dimensions = copy.deepcopy(configs), copy.deepcopy(dimensions)
    obj_path = models.choose_obj(str(obj_path))
    label = models.slug(label) if label else models.object_label(obj_path)
    xml_path = models.make_centered_xml(obj_path, label)
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "mcb_part_free")
    address = int(model.jnt_qposadr[joint_id])
    center = (finite_points([args.center_m], "--center-m")[0] if args.center_m is not None
              else [float(model.qpos0[address]), float(model.qpos0[address + 1]), 0.84])
    dimensions["object_center_world_m"] = center
    for config in configs:
        config["object_position_world_m"] = center
    seed = args.seed + (int(category_index) * 100_000 if category_index is not None else 0)
    plan = {
        "schema_version": 2, "object": label, "source_obj": str(obj_path),
        "source_obj_sha256": hashlib.sha256(obj_path.read_bytes()).hexdigest(),
        "model_sources_sha256": model_fingerprint(xml_path),
        "collector_sha256": hashlib.sha256(adapter.DEFAULT_COLLECTOR.read_bytes()).hexdigest(),
        "adapter_sha256": hashlib.sha256(Path(adapter.__file__).read_bytes()).hexdigest(),
        "sensor_guard_sha256": hashlib.sha256((HERE / "sensor_collision_guard.py").read_bytes()).hexdigest(),
        "sensor_safety": {"minimum_clearance_m": args.sensor_clearance_mm / 1000,
                          "scope": "different_fingers_gel_and_housing",
                          "rejection": "discard_entire_episode"},
        "depth_units": "normalized_heightmap_unitless",
        "seed": seed, "free_floating": args.free_floating,
        "save_rgb": True, "save_depth": True, "save_flow": True,
        "output_layout": args.layout,
        "formats": {"rgb": "png", "depth": "npz", "flow": "npz", "signals": "csv", "metadata": "json"},
        "flow_backend": "opencv_farneback_adjacent_rgb", "image_noise_sigma": 0.0,
        "dimensions": dimensions, "configs": configs,
    }
    if category is not None:
        plan.update(category=category, category_index=int(category_index))
    return {"obj_path": obj_path, "xml_path": xml_path, "label": label,
            "configs": configs, "dimensions": dimensions, "plan": plan}


def print_object_plan(args, prepared, output) -> None:
    dimensions = prepared["dimensions"]
    print(f"[PLAN] {prepared['label']}: {dimensions['full_combination_count']} combinations, "
          f"selected {len(prepared['configs'])} × repeats {args.repeats} = {dimensions['episodes']} episodes")
    print("[PLAN] world positions + object orientations + object-relative contact offsets")
    print(f"[PLAN] collection center (world meters)={dimensions['object_center_world_m']}")
    print(f"[PLAN] gripper follows object orientation={args.follow_object_orientation}; "
          "RGB/depth=on; adjacent-frame Farneback flow=on")
    print(f"[OUTPUT] {output}")
    print(f"[FORMATS] layout={args.layout}; RGB=PNG; heightmap/flow=NPZ; signals=CSV; metadata=JSON")
    print(f"[SENSOR SAFETY] minimum clearance={args.sensor_clearance_mm:g} mm; "
          "reject the whole episode before sensor assemblies meet")
    raw_bytes = 3 * (32 * 240 * 320 * (3 + 4) + 31 * 240 * 320 * 2 * 4)
    print(f"[STORAGE] {dimensions['episodes'] * raw_bytes / 1024**3:.2f} GiB before compression")


def _load_collecting_module():
    import os
    os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "mcb_taxim_numba_cache"))
    collector = adapter.load_collector(adapter.DEFAULT_COLLECTOR)
    import cv2
    if not hasattr(cv2, "calcOpticalFlowFarneback"):
        raise RuntimeError("当前 OpenCV 缺少 Farneback 光流支持")
    return collector


def summarize_object(output, label, layout, planned) -> dict:
    from collection_storage import episode_paths
    summary = {"planned": planned, "completed": 0, "valid": 0, "invalid": 0,
               "rgb_samples": 0, "flow_samples": 0, "incomplete": 0}
    for index in range(planned):
        path = episode_paths(Path(output) / label / f"episode_{index:04d}", output, layout)["metadata"]
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            metadata = {}
        if not metadata.get("collection_complete"):
            summary["incomplete"] += 1
            continue
        summary["completed"] += 1
        summary["valid" if metadata.get("valid") else "invalid"] += 1
        summary["rgb_samples"] += bool(metadata.get("tactile_images_saved"))
        summary["flow_samples"] += bool(metadata.get("optical_flow_saved"))
    return summary


def run_prepared_object(args, prepared, output, *, collector=None) -> dict:
    """Run one object into a shared batch root, keeping format folders grouped."""
    from collection_storage import episode_paths
    output = Path(output).resolve()
    collector = collector or _load_collecting_module()
    label, configs, plan = prepared["label"], prepared["configs"], prepared["plan"]
    scene_folder = output / "models"
    scene_folder.mkdir(parents=True, exist_ok=True)
    xml_path = scene_folder / prepared["xml_path"].name
    source_xml = prepared["xml_path"].read_bytes()
    if xml_path.exists():
        if xml_path.read_bytes() != source_xml:
            raise ValueError("输出目录中的仿真场景与本次模型不一致")
    else:
        xml_path.write_bytes(source_xml)
    original_step = collector.mujoco.mj_step
    try:
        adapter.configure_collector(collector, xml_path, label, output, configs,
                                    args.repeats, plan["seed"], not args.headless, True,
                                    not args.free_floating,
                                    sensor_clearance_m=args.sensor_clearance_mm / 1000)
        collector.XY_RANDOM_RANGE_M = 0.0
        collector.OBJECT_POSES = {"explicit_configs": [1.0, 0.0, 0.0, 0.0]}
        collector.POSE_STAGES = [{"name": "position_orientation_contact_sweep", "configs": configs}]
        collector.NUM_EPISODES = len(configs) * args.repeats
        collector.SAVE_OPTICAL_FLOW = True
        collector.OUTPUT_LAYOUT = args.layout
        collector.TACTILE_IMAGE_NOISE_SIGMA = 0.0
        collector.RESUME_COMPLETED_EPISODES = args.resume
        collector.EPISODE_SHARD_INDEX = 0
        collector.EPISODE_SHARD_COUNT = 1
        original_run = collector.run_episode

        def annotated_episode(*episode_args, **episode_kwargs):
            result = original_run(*episode_args, **episode_kwargs)
            metadata = result.setdefault("metadata", {})
            metadata["source_obj"] = plan["source_obj"]
            if "category" in plan:
                metadata["category"] = plan["category"]
                metadata["category_index"] = plan["category_index"]
            return result

        collector.run_episode = annotated_episode
        if args.resume:
            expanded = [cfg for stage in collector.build_pose_stages() for cfg in stage["configs"]]
            complete = True
            for index in range(collector.NUM_EPISODES):
                episode = output / label / f"episode_{index:04d}"
                path = episode_paths(episode, output, args.layout)["metadata"]
                try:
                    prior = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    complete = False
                    break
                if not collector.episode_output_complete(episode, prior, expanded[index // args.repeats]):
                    complete = False
                    break
            if complete:
                print(f"[RESUME] {label}: 全部样本完整，跳过该物体的传感器初始化。", flush=True)
                return summarize_object(output, label, args.layout, collector.NUM_EPISODES)
        print(f"[INIT] {label}: 三路 Taxim 首次背景预处理较慢，随后连续采集。", flush=True)
        collector.main()
        return summarize_object(output, label, args.layout, collector.NUM_EPISODES)
    finally:
        collector.mujoco.mj_step = original_step
        collector.TACTILE_RENDERERS.clear()


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.all_categories:
        from collect_mcb_categories import run_all_categories
        return run_all_categories(args)
    configs, dimensions = make_configs(args)
    obj_path = models.choose_obj(args.obj)
    prepared = prepare_object(args, obj_path, configs=configs, dimensions=dimensions)
    output = (args.output or DEFAULT_OUTPUT / datetime.now().strftime("%Y%m%d_%H%M%S_%f")).resolve()
    print_object_plan(args, prepared, output)
    if not args.run:
        print("[CHECK ONLY] 检查完成。加 --run 开始采集；--max-configs 2 可先试采两个组合。")
        print(json.dumps(prepared["configs"][:2], ensure_ascii=False, indent=2))
        return 0
    collector = _load_collecting_module()
    save_plan(output, prepared["plan"], args.resume)
    result = run_prepared_object(args, prepared, output, collector=collector)
    print(f"[COLLECTION RESULT] {json.dumps(result, ensure_ascii=False)}", flush=True)
    return 0 if result["valid"] and not result["incomplete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
