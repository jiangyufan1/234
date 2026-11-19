#!/usr/bin/env python
"""Collect zero-gravity MCB tactile episodes with the existing UR3/Taxim rig.

This thin adapter reuses the project's validated three-finger collector and
changes only the object-specific names, sampling plan, and Taxim mesh binding.
It is intentionally safe by default: --check-only compiles every selected MJCF;
actual collection requires --run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import operator
from pathlib import Path
import sys
import tempfile
import types
import xml.etree.ElementTree as ET

import numpy as np

from sensor_collision_guard import SensorCollisionError, SensorCollisionGuard


DEFAULT_PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_XML_DIR = DEFAULT_PROJECT / "simulation" / "mcb_xml"
DEFAULT_COLLECTOR = Path(__file__).resolve().parent / "collect_underactuated_tactile_dataset_pose_sweep_v3.py"
if not DEFAULT_COLLECTOR.is_file():
    DEFAULT_COLLECTOR = Path(__file__).resolve().parent / "collect_underactuated_tactile_dataset_pose_sweep_v3.py"
_CODE_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = DEFAULT_PROJECT / "data" / "collected"
FINGERS = ("finger1", "finger2", "finger3")


def load_collector(path: Path, skip_taxim: bool = False):
    if not path.is_file():
        raise FileNotFoundError(f"现有三指 Taxim 采集器不存在：{path}")
    os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "mcb_taxim_numba_cache"))
    spec = importlib.util.spec_from_file_location("mcb_base_tactile_collector", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载采集器：{path}")
    # The base script imports Taxim at module import time. Let physics-only
    # pilots avoid Numba/OpenCV startup when the user explicitly disables RGB.
    prior_taxim = sys.modules.get("TaximSensor")
    if skip_taxim:
        taxim_stub = types.ModuleType("TaximSensor")
        taxim_stub.TaximSensor = object
        sys.modules["TaximSensor"] = taxim_stub
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        # A physics-only probe must not replace the real renderer on a later
        # image-collecting import in the same Python process.
        if skip_taxim:
            if prior_taxim is None:
                sys.modules.pop("TaximSensor", None)
            else:
                sys.modules["TaximSensor"] = prior_taxim
    return module


def object_names(xml_path: Path) -> tuple[str, str]:
    root = ET.parse(xml_path).getroot()
    visual = root.find(".//geom[@name='mcb_part_visual']")
    if visual is None or not visual.get("mesh"):
        raise ValueError(f"{xml_path.name}: 找不到 mcb_part_visual mesh")
    return visual.get("name", "mcb_part_visual"), visual.attrib["mesh"]




def get_mcb_mesh_center_body(collector, xml_path, geom_name):
    """Return the compiled mesh AABB midpoint in object-body coordinates."""
    mj = collector.mujoco
    model = collector.load_model(Path(xml_path))

    gid = collector.name2id(
        model, mj.mjtObj.mjOBJ_GEOM, geom_name
    )
    mid = int(model.geom_dataid[gid])

    if mid < 0:
        raise RuntimeError("MCB visual geom has no mesh")

    adr = int(model.mesh_vertadr[mid])
    count = int(model.mesh_vertnum[mid])
    vertices = np.asarray(
        model.mesh_vert[adr:adr + count], dtype=float
    )

    if count <= 0 or not np.isfinite(vertices).all():
        raise RuntimeError("Invalid MCB mesh vertices")

    lo = vertices.min(axis=0)
    hi = vertices.max(axis=0)
    midpoint = (lo + hi) / 2.0

    Rm = np.zeros(9)
    mj.mju_quat2Mat(Rm, model.mesh_quat[mid])

    Rg = np.zeros(9)
    mj.mju_quat2Mat(Rg, model.geom_quat[gid])

    center_body = (
        np.asarray(model.geom_pos[gid], dtype=float)
        + Rg.reshape(3, 3) @ (
            np.asarray(model.mesh_pos[mid], dtype=float)
            + Rm.reshape(3, 3) @ midpoint
        )
    )

    return center_body, hi - lo, count


def configure_collector(collector, xml_path: Path, label: str, output: Path,
                        configs: list[dict], repeats: int, seed: int,
                        show_viewer: bool, save_images: bool,
                        pin_object: bool, sensor_clearance_m: float = 0.001) -> None:
    if not np.isfinite(sensor_clearance_m) or sensor_clearance_m <= 0:
        raise ValueError("sensor_clearance_m must be finite and greater than zero")
    visual_geom, mesh_name = object_names(xml_path)

    # Original main-branch grasp convention
    collector.GRASP_CENTER_OBJECT = np.array(
        [0.0, 0.0, 0.0],
        dtype=float,
    )

    collector.GRASP_OFFSET_TOOL = np.array(
        [0.14965, 0.0, 0.0],
        dtype=float,
    )

    print(
        "[MAIN GRASP CONVENTION]",
        {
            "object_body": "mcb_part",
            "grasp_center_object_m":
                collector.GRASP_CENTER_OBJECT.tolist(),
            "tcp_site": "tool0",
            "tcp_offset_tool_m":
                collector.GRASP_OFFSET_TOOL.tolist(),
        },
        flush=True,
    )

    collector.XML_PATH = xml_path
    collector.OBJECT_BODY = "mcb_part"
    collector.OBJECT_FREEJOINT = "mcb_part_free"
    collector.DATASET_OBJECT_LABEL = label
    collector.RUN_OUTPUT_ROOT = output
    collector.SHOW_VIEWER = show_viewer
    collector.SAVE_GELSIGHT_IMAGES = save_images
    # MCB shapes may make only one or two pads touch at first. Keep those
    # useful labeled observations; contact masks per finger remain in signals.
    collector.REQUIRE_ALL_THREE_CONTACT = False
    collector.BASE_SEED = seed
    collector.REPEATS_PER_CONFIG = repeats
    collector.NUM_EPISODES = len(configs) * repeats
    # Zero gravity is part of this experiment's definition.
    collector.OBJECT_DROP_CENTER_Z = 0.84
    collector.OBJECT_SETTLE_TIME = 0.0
    # A 120 mm retreat puts some default MCB targets beyond the UR3 workspace.
    # Keep a 100 mm safety pose and a separate 55 mm pregrasp pose.
    collector.SAFE_DISTANCE = 0.100
    collector.PREGRASP_DISTANCE = 0.055
    collector.XY_RANDOM_RANGE_M = 0.002
    collector.OBJECT_POSES = {"identity": [1.0, 0.0, 0.0, 0.0]}
    collector.OBJECT_POSE_DROP_Z = {"identity": 0.84}
    collector.POSE_STAGES = [{"name": "mcb_contact_sweep", "configs": configs}]
    collector.TACTILE_RENDERERS.clear()

    # In microgravity, an unrestrained part recoils when the tactile pads press
    # it. The default virtual fixture models the astronaut's other hand holding
    # the part steady; pass --free-floating to collect object motion instead.
    lock = {"model": None, "qpos": None, "qadr": None, "dadr": None}
    if not hasattr(collector, "_mcb_base_mj_step"):
        collector._mcb_base_mj_step = collector.mujoco.mj_step
        collector._mcb_base_randomize = collector.randomize_object_pose
        collector._mcb_base_run_episode = collector.run_episode
        collector._mcb_base_reset_home = collector.reset_home
    original_step = collector._mcb_base_mj_step
    original_randomize = collector._mcb_base_randomize

    def reset_without_prior_fixture(model, data):
        # Each episode starts from the XML reference pose. Reusing the previous
        # lock here would change the next episode's reference orientation.
        if lock["model"] is model:
            lock.update(model=None, qpos=None, qadr=None, dadr=None)
        collector._active_model = model
        return collector._mcb_base_reset_home(model, data)

    collector.reset_home = reset_without_prior_fixture

    if pin_object:
        def locked_step(model, data, *step_args, **step_kwargs):
            result = original_step(model, data, *step_args, **step_kwargs)
            if model is lock["model"] and lock["qpos"] is not None:
                data.qpos[lock["qadr"]:lock["qadr"] + 7] = lock["qpos"]
                data.qvel[lock["dadr"]:lock["dadr"] + 6] = 0.0
                collector.mujoco.mj_forward(model, data)
            return result

        def fixed_randomize(model, data, rng, pose_cfg):
            result = original_randomize(model, data, rng, pose_cfg)
            jid = collector.name2id(model, collector.mujoco.mjtObj.mjOBJ_JOINT,
                                    collector.OBJECT_FREEJOINT)
            qadr = int(model.jnt_qposadr[jid])
            dadr = int(model.jnt_dofadr[jid])
            lock.update(model=model, qpos=data.qpos[qadr:qadr + 7].copy(),
                        qadr=qadr, dadr=dadr)
            return result

        collector.mujoco.mj_step = locked_step
        collector.randomize_object_pose = fixed_randomize
    else:
        collector.mujoco.mj_step = original_step
        collector.randomize_object_pose = original_randomize

    original_run_episode = collector._mcb_base_run_episode

    def record_fixture(*run_args, **run_kwargs):
        result = original_run_episode(*run_args, **run_kwargs)
        metadata = result.setdefault("metadata", {})
        contact_frames = sum(
            any(sample.get("contacts", []))
            for sample in result.get("samples", [])
        )
        metadata["contact_frame_count"] = int(contact_frames)
        metadata["virtual_fixture_holds_object"] = bool(pin_object)
        if result.get("valid", False) and contact_frames < 3:
            result["valid"] = False
            result["reason"] = "fewer_than_3_contact_frames"
        metadata["valid"] = bool(result["valid"])
        metadata["reason"] = result["reason"]
        return result

    collector.run_episode = record_fixture

    # The MCB XML uses three independent position actuators, while the older
    # YCB collector expects one tendon actuator. Adapt its scalar open/close
    # command to the three finger targets and calibrate direction by pad spread.
    collector.FINGER_ACTUATORS = ["finger1_ctrl", "finger2_ctrl", "finger3_ctrl"]
    collector.GRIPPER_ACTUATOR = collector.FINGER_ACTUATORS[0]

    def set_three_fingers(data, gripper_act_id, value):
        # Resolve IDs from ctrl addresses at runtime (all three actuators are
        # position actuators with one scalar control each).
        model = collector._active_model
        target = float(np.clip(value / collector.CLOSE_CTRL_MAG * 0.5, -0.5, 0.5))
        for actuator_name in collector.FINGER_ACTUATORS:
            aid = collector.name2id(model, collector.mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
            data.ctrl[aid] = target

    collector.set_gripper_ctrl = set_three_fingers

    def spread_for_ctrl(xml_path_arg, ctrl_value):
        load_model = getattr(collector, "load_model", collector.mujoco.MjModel.from_xml_path)
        model = load_model(str(xml_path_arg))
        data = collector.mujoco.MjData(model)
        collector.reset_home(model, data)
        arm_ids = collector.get_actuator_ids(model, collector.ARM_ACTUATORS)
        arm_qpos, _ = collector.get_joint_addresses(model, collector.ARM_JOINTS)
        q_home = collector.get_arm_q(data, arm_qpos)
        finger_ids = [collector.name2id(model, collector.mujoco.mjtObj.mjOBJ_ACTUATOR, n)
                      for n in collector.FINGER_ACTUATORS]
        collector._active_model = model
        for _ in range(int(collector.CALIBRATION_TIME / model.opt.timestep)):
            collector.hold_arm(data, arm_ids, q_home)
            target = float(np.clip(ctrl_value / collector.CLOSE_CTRL_MAG * 0.5, -0.5, 0.5))
            data.ctrl[finger_ids] = target
            collector.mujoco.mj_step(model, data)
        collector.mujoco.mj_forward(model, data)
        return collector.mean_gel_spread(model, data)

    collector.test_gripper_ctrl_spread = spread_for_ctrl

    def infer_close_sign(xml_path_arg):
        positive = spread_for_ctrl(xml_path_arg, collector.CALIBRATION_CTRL_MAG)
        negative = spread_for_ctrl(xml_path_arg, -collector.CALIBRATION_CTRL_MAG)
        sign = 1 if positive < negative else -1
        print(f"[GRIPPER CALIBRATION] + spread={positive * 1000:.2f} mm, "
              f"- spread={negative * 1000:.2f} mm; close_sign={sign:+d}")
        return sign

    collector.determine_close_sign = infer_close_sign

    def init_mcb_renderers(model, data):
        collector.TACTILE_RENDERERS.clear()
        collector._active_model = model
        if not collector.SAVE_GELSIGHT_IMAGES:
            return
        for finger in FINGERS:
            sensor = collector.TaximSensor(
                sensor_type=collector.TAXIM_SENSOR_TYPE,
                bg_index=0,
                resize=None,
                preprocess_bg=True,
            )
            sensor.add_geom_mujoco(
                visual_geom, model=model, data=data, mesh_name=mesh_name
            )
            sensor.add_camera_mujoco(f"{finger}_gelsight_site", model, data)
            sensor.set_sensor_pad_geom(f"{finger}_gel")
            collector.TACTILE_RENDERERS[finger] = sensor

    collector.init_tactile_renderers = init_mcb_renderers
    install_sensor_safety(collector, sensor_clearance_m, pin_object=pin_object)


def install_sensor_safety(collector, clearance_m=0.001, *, pin_object=False):
    """Stop before sensor assemblies meet; reject the entire affected episode.

    Collision masks alone allow pads to pass through one another. This guard
    measures their geometry independently, including the sensor housings.
    Calibration is outside the episode scope and produces no saved samples.
    """
    if not np.isfinite(clearance_m) or clearance_m <= 0:
        raise ValueError("Sensor clearance must be finite and greater than zero")
    original_step = collector.mujoco.mj_step
    original_run = collector.run_episode
    original_complete = getattr(collector, "_mcb_base_episode_output_complete", None)
    if original_complete is None and hasattr(collector, "episode_output_complete"):
        original_complete = collector.episode_output_complete
        collector._mcb_base_episode_output_complete = original_complete
    policy = {"enabled": True, "minimum_clearance_m": float(clearance_m),
              "scope": "different_fingers_gel_and_housing",
              "rejection": "discard_entire_episode", "version": 1,
              "distance_method": "obb_separating_axis_lower_bound",
              "guard_sha256": hashlib.sha256((Path(__file__).resolve().parent /
                                               "sensor_collision_guard.py").read_bytes()).hexdigest()}
    collector.SENSOR_SAFETY_POLICY = policy
    state = {"active": False}

    def inspect(model, data):
        if state.get("model") is not model:
            state["guard"] = SensorCollisionGuard(model, clearance_m=clearance_m)
            state["model"] = model
        report = state["guard"].inspect(data)
        state["checks"] += 1
        value = report["minimum_distance_m"]
        distance = float(value) if value is not None else float("inf")
        if np.isfinite(distance) and distance < state["minimum_distance_m"]:
            state["minimum_distance_m"] = distance
            state["closest_pair"] = report["closest_pair"]
        if not report["safe"]:
            state["violation"] = report
            raise SensorCollisionError(report)

    def guarded_step(model, data, *step_args, **step_kwargs):
        if not state["active"]:
            return original_step(model, data, *step_args, **step_kwargs)
        if len(step_args) > 1 or (step_args and "nstep" in step_kwargs):
            raise TypeError("mj_step received conflicting nstep arguments")
        options = dict(step_kwargs)
        count = operator.index(step_args[0] if step_args else options.pop("nstep", 1))
        if options or count < 0:
            raise TypeError("Unsupported mj_step arguments")
        spec = collector.mujoco.mjtState.mjSTATE_INTEGRATION
        if state.get("snapshot_model") is not model:
            state["snapshot"] = np.empty(collector.mujoco.mj_stateSize(model, spec))
            state["snapshot_model"] = model
        # Check even nstep=0, and split bulk steps so no intermediate collision
        # can disappear before the final state is inspected.
        collector.mujoco.mj_kinematics(model, data)
        inspect(model, data)
        for _ in range(count):
            collector.mujoco.mj_getState(model, data, state["snapshot"], spec)
            original_step(model, data)
            collector.mujoco.mj_kinematics(model, data)
            try:
                inspect(model, data)
            except SensorCollisionError:
                collector.mujoco.mj_setState(model, data, state["snapshot"], spec)
                collector.mujoco.mj_forward(model, data)
                state["restored_safe_state"] = True
                raise
            state["steps"] += 1

    def guarded_episode(*episode_args, **episode_kwargs):
        state.clear()
        state.update(active=True, model=None, checks=0, steps=0,
                     minimum_distance_m=float("inf"), closest_pair=None,
                     violation=None, restored_safe_state=False)
        try:
            result = original_run(*episode_args, **episode_kwargs)
        except SensorCollisionError:
            pose = episode_kwargs.get("pose_cfg", episode_args[4] if len(episode_args) > 4 else {})
            index = episode_kwargs.get("episode_idx", episode_args[3] if len(episode_args) > 3 else 0)
            result = {"valid": False, "reason": "sensor_clearance_violation", "samples": [],
                      "metadata": {"pose_config": pose, "episode_index": int(index),
                                   "virtual_fixture_holds_object": bool(pin_object)}}
        finally:
            state["active"] = False
        if state["violation"] is not None:
            # Also covers an inner caller that catches the guard exception.
            result.update(valid=False, reason="sensor_clearance_violation", samples=[])
        elif result.get("samples") and not state["checks"]:
            raise RuntimeError("Refusing samples without sensor clearance checks")
        metadata = result.setdefault("metadata", {})
        minimum = state["minimum_distance_m"]
        metadata["sensor_safety"] = {
            "policy": policy, "passed": state["violation"] is None and state["checks"] > 0,
            "checks": state["checks"], "physics_steps": state["steps"],
            "minimum_distance_m": None if not np.isfinite(minimum) else minimum,
            "distance_is_lower_bound": True,
            "closest_pair": state["closest_pair"], "violation": state["violation"],
            "restored_safe_state": state["restored_safe_state"],
            "discarded_entire_episode": state["violation"] is not None,
        }
        metadata["valid"], metadata["reason"] = bool(result["valid"]), result["reason"]
        if state["violation"] is not None:
            metadata["contact_frame_count"] = 0
            metadata["tactile_images_saved"] = False
            metadata["optical_flow_saved"] = False
        return result

    collector.mujoco.mj_step, collector.run_episode = guarded_step, guarded_episode
    if hasattr(collector, "capture_gelsight_images"):
        if not hasattr(collector, "_mcb_base_capture_gelsight_images"):
            collector._mcb_base_capture_gelsight_images = collector.capture_gelsight_images
        def guarded_capture(model, data):
            if state["active"]:
                collector.mujoco.mj_kinematics(model, data)
                inspect(model, data)
            images = collector._mcb_base_capture_gelsight_images(model, data)
            if state["active"]:
                collector.mujoco.mj_kinematics(model, data)
                inspect(model, data)
            return images
        collector.capture_gelsight_images = guarded_capture
    if original_complete is not None:
        def safe_complete(episode, metadata, pose_cfg):
            safety = metadata.get("sensor_safety", {})
            if safety.get("policy") != policy:
                return False
            if metadata.get("valid") and safety.get("passed") is not True:
                return False
            return original_complete(episode, metadata, pose_cfg)
        collector.episode_output_complete = safe_complete


def make_configs(args) -> list[dict]:
    if args.grid:
        # Object-relative offsets move the three-pad contact footprint over
        # the center and upper/lower regions while keeping the first pilot small.
        z_offsets = (-0.012, 0.0, 0.012)
        xy_offsets = ((0.0, 0.0), (-0.008, 0.0), (0.008, 0.0),
                      (0.0, -0.008), (0.0, 0.008))
        offsets = [(x, y, z) for z in z_offsets for x, y in xy_offsets]
    else:
        offsets = [(0.0, 0.0, 0.0)]
    offsets = offsets[:args.max_configs]
    return [
        {
            "grasp_offset_object_m": [float(v) for v in offset],
            "approach_azimuth_deg": 0.0,
            "approach_tilt_deg": float(tilt),
            "gripper_roll_deg": float(roll),
        }
        for offset in offsets
        for tilt in args.tilts
        for roll in args.rolls
    ]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml-dir", type=Path, default=DEFAULT_XML_DIR)
    parser.add_argument("--collector", type=Path, default=DEFAULT_COLLECTOR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--objects", nargs="*", help="XML stem(s); default: every XML in --xml-dir")
    parser.add_argument("--check-only", action="store_true", help="compile/check XMLs, no data collection")
    parser.add_argument("--run", action="store_true", help="run the tactile collector")
    parser.add_argument("--grid", action="store_true", help="enable spatial offset grid")
    parser.add_argument("--max-configs", type=int, default=1, help="cap position offsets; default pilot=1")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--rolls", type=float, nargs="+", default=[0.0])
    parser.add_argument("--tilts", type=float, nargs="+", default=[0.0])
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--show-viewer", action="store_true")
    parser.add_argument("--no-images", action="store_true", help="save contacts/metadata only; skip Taxim rendering")
    parser.add_argument("--free-floating", action="store_true", help="allow the part to drift when contacted")
    parser.add_argument("--sensor-clearance-mm", type=float, default=1.0,
                        help="reject a whole episode before different finger sensors get this close")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.run == args.check_only:
        raise SystemExit("请且仅指定一个：--check-only 或 --run")
    if args.repeats < 1 or args.max_configs < 1:
        raise SystemExit("--repeats 和 --max-configs 必须至少为 1")
    if not np.isfinite(args.sensor_clearance_mm) or args.sensor_clearance_mm <= 0:
        raise SystemExit("--sensor-clearance-mm 必须为大于0的有限数值")
    xml_paths = sorted(args.xml_dir.glob("*.xml"))
    if args.objects:
        wanted = set(args.objects)
        xml_paths = [p for p in xml_paths if p.stem in wanted or p.name in wanted]
        found = {p.stem for p in xml_paths} | {p.name for p in xml_paths}
        missing = wanted - found
        if missing:
            raise SystemExit(f"未找到指定模型：{', '.join(sorted(missing))}")
    if not xml_paths:
        raise SystemExit(f"没有找到 XML：{args.xml_dir}")

    import mujoco

    compiled = []
    for path in xml_paths:
        model = mujoco.MjModel.from_xml_path(str(path))
        if not np.allclose(model.opt.gravity, 0.0, atol=1e-12):
            raise RuntimeError(f"{path.name}: gravity 不是 0：{model.opt.gravity}")
        for kind, name in ((mujoco.mjtObj.mjOBJ_BODY, "mcb_part"),
                           (mujoco.mjtObj.mjOBJ_JOINT, "mcb_part_free"),
                           (mujoco.mjtObj.mjOBJ_GEOM, "mcb_part_collision")):
            if mujoco.mj_name2id(model, kind, name) < 0:
                raise RuntimeError(f"{path.name}: 缺少 MuJoCo 对象 {name}")
        object_names(path)
        compiled.append(path)
        print(f"[OK] {path.name}: gravity={model.opt.gravity.tolist()}, ngeom={model.ngeom}")
    print(f"模型编译检查通过：{len(compiled)} 个。")
    if args.check_only:
        return 0

    configs = make_configs(args)
    if not configs:
        raise SystemExit("采集配置为空")
    args.output.mkdir(parents=True, exist_ok=True)
    # Taxim's @njit(cache=True) kernel writes a cache at import/compile time.
    # Put it under the writable dataset root (the model/CAD tree can be read-only).
    numba_cache = args.output / ".numba_cache"
    numba_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(numba_cache))
    collector = load_collector(args.collector, skip_taxim=args.no_images)
    run_root = args.output / "runs"
    run_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "collection_plan.csv"
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["object", "xml", "episodes", "configs", "gravity"])
        for p in compiled:
            writer.writerow([p.stem, str(p), len(configs) * args.repeats,
                             json.dumps(configs, ensure_ascii=False), "0 0 0"])

    for index, xml_path in enumerate(compiled):
        label = xml_path.stem
        output = run_root / label
        print(f"\n=== {label}: {len(configs) * args.repeats} episodes ===", flush=True)
        configure_collector(
            collector, xml_path, label, output, configs,
            args.repeats, args.seed + index * 100_000,
            args.show_viewer, not args.no_images, not args.free_floating,
            sensor_clearance_m=args.sensor_clearance_mm / 1000,
        )
        collector.main()
    print(f"完成。采集计划和数据位于：{args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
