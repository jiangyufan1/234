from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COLLECTION_DIR = PROJECT_ROOT / "scripts" / "collection"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(COLLECTION_DIR) not in sys.path:
    sys.path.insert(0, str(COLLECTION_DIR))

# Make direct execution from the project root reliable on Windows.
# collector.py lives at <project>/scripts/collection/event_driven_v2/.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from scripts.collection.event_driven_v2.core import (
        Config, ContactTracker, EventBuffer, GripperController,
        State, StateMachine, assess
    )
except ModuleNotFoundError as exc:
    # Also support running/importing this file directly from its own folder.
    if exc.name not in {"scripts", "scripts.collection", "scripts.collection.event_driven_v2"}:
        raise
    from core import (
        Config, ContactTracker, EventBuffer, GripperController,
        State, StateMachine, assess
    )


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def preflight(xml_path):
    import mujoco
    xml_path = xml_path.resolve()
    if not xml_path.is_file():
        raise FileNotFoundError(xml_path)
    root = ET.parse(xml_path).getroot()
    required = {
        "body": ["mcb_part"],
        "freejoint": ["mcb_part_free"],
        "geom": ["mcb_part_visual", "mcb_part_collision",
                 "finger1_gel", "finger2_gel", "finger3_gel"],
        "actuator": ["finger1_ctrl", "finger2_ctrl", "finger3_ctrl"],
        "site": ["tool0", "finger1_gelsight_site", "finger2_gelsight_site",
                 "finger3_gelsight_site"],
    }
    for tag, names in required.items():
        if tag == "actuator":
            actuator_root = root.find("actuator")
            present = (
                {node.get("name") for node in actuator_root.iter()
                 if node.get("name")}
                if actuator_root is not None else set()
            )
        else:
            present = {
                node.get("name") for node in root.iter(tag)
                if node.get("name")
            }
        missing = sorted(set(names) - present)
        if missing:
            raise ValueError(f"XML missing {tag}: {missing}")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    if any(abs(float(x)) > 1e-12 for x in model.opt.gravity):
        raise ValueError(f"expected zero gravity, got {model.opt.gravity}")
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "mcb_part_free")
    if jid < 0 or model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError("mcb_part_free is not a free joint")
    return {
        "xml": str(xml_path), "gravity": model.opt.gravity.tolist(),
        "timestep_s": float(model.opt.timestep), "nq": int(model.nq),
        "nv": int(model.nv), "ngeom": int(model.ngeom), "nu": int(model.nu),
        "object_free_joint": True,
        "visual_mesh": root.find(".//geom[@name='mcb_part_visual']").get("mesh"),
    }


def install_event_patch(collector, cfg):
    def collect_contact_sequence(
        model, data, viewer, arm_act_ids, q_arm_hold, gripper_act_id,
        finger_qpos_adr, gel_geom_ids, object_geom_ids, open_ctrl,
        close_ctrl_limit, close_sign,
    ):
        dt = float(model.opt.timestep)
        sample_every = max(1, int(round(1.0 / (cfg.capture_hz * dt))))
        first_timeout = int(cfg.max_wait_first_s / dt)
        three_timeout = int(cfg.max_wait_three_s / dt)
        grace_steps = int(cfg.close_limit_grace_s / dt)

        tracker = ContactTracker(dt, cfg.stable_time_s)
        buffer = EventBuffer(cfg.precontact_frames)
        sm = StateMachine()

        # MCB adapter exposes three independent position actuators. Its legacy
        # scalar command is mapped linearly to actuator target [-0.5, +0.5].
        finger_names = ("finger1_ctrl", "finger2_ctrl", "finger3_ctrl")
        finger_act_ids = [
            collector.name2id(model, collector.mujoco.mjtObj.mjOBJ_ACTUATOR, n)
            for n in finger_names
        ]
        scalar_commands = [float(open_ctrl)] * 3

        def scalar_to_target(value):
            return float(max(-0.5, min(0.5,
                value / float(collector.CLOSE_CTRL_MAG) * 0.5)))

        def apply_finger_targets():
            for i, aid in enumerate(finger_act_ids):
                data.ctrl[aid] = scalar_to_target(scalar_commands[i])

        first_step = None
        stable = False
        hold_lost = False
        limit_steps = [None, None, None]
        close_start = float(data.time)
        close_stop = None
        rates_log = []
        last_rate_tuple = None
        failure = None
        step = 0

        while True:
            collector.hold_arm(data, arm_act_ids, q_arm_hold)

            # Read the latest contact state independently for each finger.
            prior_states = collector.get_finger_contact_state(
                model, data, gel_geom_ids, object_geom_ids
            )
            prior_bits = [bool(x["contact"]) for x in prior_states]
            contact_count = sum(prior_bits)

            # A finger that is in contact holds its own target. Uncontacted
            # fingers continue closing; the remaining fingers slow as more
            # fingers make contact.
            rate = (cfg.rate_0, cfg.rate_1, cfg.rate_2, cfg.rate_3)[contact_count]
            for i in range(3):
                if not prior_bits[i] and not stable:
                    next_value = scalar_commands[i] + int(close_sign) * rate * dt
                    if close_sign < 0:
                        scalar_commands[i] = max(next_value, float(close_ctrl_limit))
                    else:
                        scalar_commands[i] = min(next_value, float(close_ctrl_limit))

                if scalar_commands[i] == float(close_ctrl_limit):
                    if limit_steps[i] is None:
                        limit_steps[i] = step

            rate_tuple = tuple(
                0.0 if prior_bits[i] or stable else float(rate)
                for i in range(3)
            )
            if rate_tuple != last_rate_tuple:
                rates_log.append({
                    "time_s": float(data.time),
                    "contact_bits": prior_bits,
                    "global_contact_count": int(contact_count),
                    "per_finger_command_rate": list(rate_tuple),
                    "scalar_commands": list(scalar_commands),
                })
                last_rate_tuple = rate_tuple

            apply_finger_targets()
            collector.mujoco.mj_step(model, data)
            collector.sync_viewer(viewer)

            states = collector.get_finger_contact_state(
                model, data, gel_geom_ids, object_geom_ids
            )
            obs = tracker.update(states, float(data.time))

            if tracker.first_time is not None and first_step is None:
                first_step = step
                sm.set(State.DYNAMIC)
                print(
                    f"[EVENT] first={tracker.first_finger} "
                    f"t={data.time:.4f}s", flush=True
                )

            if obs["count"] == 3 and not stable:
                if sm.state == State.DYNAMIC:
                    sm.set(State.CONFIRM)
            elif obs["count"] < 3 and sm.state == State.CONFIRM:
                sm.set(State.DYNAMIC)

            if obs["stable"] and not stable:
                stable = True
                close_stop = float(data.time)
                sm.set(State.HOLD)
                print(
                    f"[EVENT] stable three-finger contact "
                    f"t={data.time:.4f}s", flush=True
                )

            if sm.state == State.HOLD and not obs["all"]:
                hold_lost = True

            if step % sample_every == 0:
                if tracker.first_time is None:
                    buffer.add_pre(collector.make_sample(
                        model, data, finger_qpos_adr, gripper_act_id,
                        states, phase="precontact"
                    ))
                elif not buffer.started:
                    buffer.start_dynamic(collector.make_sample(
                        model, data, finger_qpos_adr, gripper_act_id,
                        states, phase="dynamic_contact"
                    ))
                elif stable:
                    buffer.add_hold(collector.make_sample(
                        model, data, finger_qpos_adr, gripper_act_id,
                        states, phase="hold"
                    ))
                else:
                    buffer.add_dynamic(collector.make_sample(
                        model, data, finger_qpos_adr, gripper_act_id,
                        states, phase="dynamic_contact"
                    ))

                if stable and buffer.hold_frames >= cfg.hold_frames:
                    sm.set(State.DONE)
                    break

            all_at_limit = all(
                (v <= float(close_ctrl_limit) + 1e-12)
                if close_sign < 0 else
                (v >= float(close_ctrl_limit) - 1e-12)
                for v in scalar_commands
            )

            if tracker.first_time is None:
                if step >= first_timeout:
                    failure = "no_first_contact_timeout"
                    break
                if all_at_limit:
                    failure = "close_limit_without_contact"
                    break
            else:
                if not stable and step - first_step >= three_timeout:
                    # Preserve this episode as partial-contact data. Do not
                    # reject it merely because a small object cannot touch all
                    # three GelSight pads.
                    failure = "partial_contact_timeout"
                    break
                # Once every still-uncontacted finger has exhausted its
                # remaining travel, give contact dynamics a short grace period.
                if not stable and all_at_limit:
                    latest_limit = max(x for x in limit_steps if x is not None)
                    if step - latest_limit >= grace_steps:
                        failure = "partial_contact_close_limit"
                        break

            step += 1

        final_ctrl = float(sum(scalar_commands) / 3.0)

        tracker.hold_lost = hold_lost
        samples = buffer.finish()
        events = tracker.as_dict()
        valid, reason = assess(events, samples, cfg.hold_frames)

        if failure and tracker.first_time is not None and not stable:
            sm.set(State.PARTIAL)
            valid = True
            reason = failure
        elif failure:
            sm.set(State.FAILED)
            valid = False
            reason = failure

        collector._event_driven_last_metadata = {
            "collection_version": "event_driven_v2",
            "object_fixed": False,
            "success_criterion": "three_finger_stable_contact",
            "episode_status": (
                "success" if reason == "success" else
                "partial_contact" if tracker.first_time is not None else
                "no_contact"
            ),
            "training_eligible": bool(valid and tracker.first_time is not None),
            "dynamic_start": "first_contact_event",
            "control_coordinate": "legacy_scalar_tendon_target_proxy",
            "control_rate_unit": "command_coordinate_rad_per_s",
            "config": cfg.as_dict(),
            "contact_events": events,
            "state_history": sm.history,
            "dynamic_frames": buffer.dynamic_frames,
            "hold_frames": buffer.hold_frames,
            "close_start_time_s": close_start,
            "close_stop_time_s": close_stop,
            "close_duration_s": None if close_stop is None else close_stop - close_start,
            "final_scalar_commands": list(scalar_commands),
            "finger_actuator_names": list(finger_names),
            "command_rate_log": rates_log,
            "valid": bool(valid),
            "reason": reason,
        }
        return {
            "valid": bool(valid), "reason": reason, "samples": samples,
            "final_ctrl": final_ctrl, "all_contact": bool(stable),
        }

    collector.collect_contact_sequence = collect_contact_sequence
    original_run = collector.run_episode

    def run_with_event_metadata(*args, **kwargs):
        result = original_run(*args, **kwargs)
        result.setdefault("metadata", {})["event_driven_v2"] = dict(
            getattr(collector, "_event_driven_last_metadata", {})
        )
        result["metadata"]["event_driven_v2"]["protocol_signature"] = getattr(
            collector, "_event_protocol_signature", None
        )
        result["metadata"]["collection_version"] = "event_driven_v2"
        return result

    collector.run_episode = run_with_event_metadata


def parse_args():
    p = argparse.ArgumentParser(description="Event-driven three-finger GelSight MCB collector V2")
    p.add_argument("--xml", type=Path,
                   default=PROJECT_ROOT / "simulation/mcb_xml/clamp_train_00015149.xml")
    p.add_argument("--collector", type=Path,
                   default=COLLECTION_DIR / "collect_underactuated_tactile_dataset_pose_sweep_v3.py")
    p.add_argument("--output", type=Path,
                   default=Path(r"E:\project\p1\deepgelsight\event_driven_v2"))
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=20261118)
    p.add_argument("--show-viewer", action="store_true")
    p.add_argument("--no-images", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--check-only", action="store_true")
    p.add_argument("--run", action="store_true")
    p.add_argument("--sensor-clearance-mm", type=float, default=1.0)
    p.add_argument("--max-wait-first-contact", type=float, default=6.0)
    p.add_argument("--max-wait-three-contact", type=float, default=6.0)
    p.add_argument("--rate-zero-contact", type=float, default=0.35)
    p.add_argument("--rate-one-contact", type=float, default=0.12)
    p.add_argument("--rate-two-contact", type=float, default=0.06)
    p.add_argument("--stable-contact-time", type=float, default=0.04)
    return p.parse_args()


def main():
    args = parse_args()
    if args.check_only == args.run:
        raise SystemExit("Specify exactly one of --check-only or --run")
    if args.episodes < 1:
        raise SystemExit("--episodes must be >= 1")

    report = preflight(args.xml)
    print("[XML PREFLIGHT]", json.dumps(report, indent=2), flush=True)
    if args.check_only:
        print("CHECK ONLY: no collection started.")
        return 0

    cfg = Config(
        max_wait_first_s=args.max_wait_first_contact,
        max_wait_three_s=args.max_wait_three_contact,
        rate_0=args.rate_zero_contact,
        rate_1=args.rate_one_contact,
        rate_2=args.rate_two_contact,
        stable_time_s=args.stable_contact_time,
    )
    os.environ["TACTILE_RESUME"] = "1" if args.resume else "0"
    mcb = load_module("event_driven_mcb_adapter",
                      COLLECTION_DIR / "collect_mcb_gelsight_dataset.py")
    collector = mcb.load_collector(args.collector.resolve(), skip_taxim=args.no_images)
    install_event_patch(collector, cfg)

    label = args.xml.stem
    configs = [{
        "grasp_offset_object_m": [0.0, 0.0, 0.0],
        "approach_azimuth_deg": 0.0,
        "approach_tilt_deg": 0.0,
        "gripper_roll_deg": 0.0,
    }]
    mcb.configure_collector(
        collector, args.xml.resolve(), label, args.output.resolve(),
        configs, args.episodes, args.seed, args.show_viewer,
        not args.no_images, False, args.sensor_clearance_mm / 1000.0
    )

    # Explicitly retain the validated legacy TCP/object-centre geometry.
    # The legacy run_episode() computes:
    # target_grasp_center_world = object_pos + object_R @ (
    #     GRASP_CENTER_OBJECT + grasp_offset_object_m
    # )
    # target_tool_pos = target_grasp_center_world - target_tool_R @ GRASP_OFFSET_TOOL
    # Do not replace these with a world-origin or object-AABB guess.
    import numpy as np
    if not hasattr(collector, "TOOL_SITE") or not hasattr(collector, "GRASP_OFFSET_TOOL"):
        raise RuntimeError("Legacy TCP/grasp offset definitions are missing")
    if not hasattr(collector, "GRASP_CENTER_OBJECT"):
        raise RuntimeError("Legacy object-centre grasp definition is missing")
    if collector.TOOL_SITE != "tool0":
        raise RuntimeError(f"Unexpected TCP site: {collector.TOOL_SITE}")
    collector.GRASP_OFFSET_TOOL = np.asarray(
        collector.GRASP_OFFSET_TOOL, dtype=float
    ).copy()
    collector.GRASP_CENTER_OBJECT = np.asarray(
        collector.GRASP_CENTER_OBJECT, dtype=float
    ).copy()
    if collector.GRASP_OFFSET_TOOL.shape != (3,) or collector.GRASP_CENTER_OBJECT.shape != (3,):
        raise RuntimeError("TCP/object-centre offsets must be 3D vectors")

    # The MCB adapter's legacy record_fixture wrapper rejects episodes with
    # fewer than three contact frames. V2 explicitly retains any episode with
    # at least one real finger contact, so normalize the result after that
    # adapter wrapper has run. No-contact episodes remain audit-only.
    _adapter_run_episode = collector.run_episode

    def _v2_classify_run_episode(*run_args, **run_kwargs):
        result = _adapter_run_episode(*run_args, **run_kwargs)
        samples = result.get("samples", [])
        contact_frames = sum(
            any(bool(x) for x in sample.get("contacts", []))
            for sample in samples
        )
        first_contact = any(
            any(bool(x) for x in sample.get("contacts", []))
            for sample in samples
        )
        stable_three = bool(result.get("all_contact", False))
        if first_contact:
            result["valid"] = True
            result["reason"] = (
                "success" if stable_three else "partial_contact"
            )
        else:
            result["valid"] = False
            result["reason"] = "no_contact"
        metadata = result.setdefault("metadata", {})
        metadata["contact_frame_count"] = int(contact_frames)
        metadata["episode_status"] = result["reason"]
        metadata["training_eligible"] = bool(first_contact)
        metadata["valid"] = bool(result["valid"])
        metadata["reason"] = result["reason"]
        metadata["all_three_contact_stable"] = stable_three
        metadata["tcp_site"] = collector.TOOL_SITE
        metadata["tcp_to_grasp_center_tool_m"] = collector.GRASP_OFFSET_TOOL.tolist()
        metadata["grasp_center_object_m"] = collector.GRASP_CENTER_OBJECT.tolist()
        metadata["grasp_offset_object_m"] = list(
            metadata.get("pose_config", {}).get("grasp_offset_object_m", [0.0, 0.0, 0.0])
        )
        metadata["positioning_method"] = "legacy_object_frame_grasp_center_plus_tcp_offset"
        return result

    collector.run_episode = _v2_classify_run_episode

    collector.OUTPUT_LAYOUT = "by_type"
    collector.EXPECTED_TOTAL_FRAMES = "variable (event-driven)"
    # Keep partial-contact episodes as usable data. Three-finger stability
    # remains a success label, not an episode-admission requirement.
    collector.REQUIRE_ALL_THREE_CONTACT = False
    collector.SAVE_OPTICAL_FLOW = False
    collector.CAPTURE_HZ = cfg.capture_hz
    collector.PRECONTACT_FRAMES = cfg.precontact_frames
    collector.HOLD_FRAMES = cfg.hold_frames
    collector.RESUME_COMPLETED_EPISODES = bool(args.resume)
    collector.RUN_OUTPUT_ROOT = args.output.resolve()
    collector.NUM_EPISODES = args.episodes

    signature = {
        "collection_version": "event_driven_v2",
        "config": cfg.as_dict(),
        "xml_sha256": hashlib.sha256(args.xml.resolve().read_bytes()).hexdigest(),
    }
    collector._event_protocol_signature = signature
    prior_complete = collector.episode_output_complete

    def protocol_aware_complete(episode_dir, metadata, pose_cfg):
        if metadata.get("event_driven_v2", {}).get("protocol_signature") != signature:
            return False
        return prior_complete(episode_dir, metadata, pose_cfg)

    collector.episode_output_complete = protocol_aware_complete

    print("[V2] free object; first-contact event capture; "
          "success=stable-three-contact; partial-contact episodes retained", flush=True)
    print("[V2] output:", args.output.resolve(), flush=True)
    collector.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
