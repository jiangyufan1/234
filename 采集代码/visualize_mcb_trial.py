#!/usr/bin/env python
"""Run a small real collection with a live viewer and a recorded scene video."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import time

import numpy as np

import collect_mcb_position_pose as core


class SceneRecorder:
    """Record actual simulator states; the saved tactile data stays untouched."""

    def __init__(self, collector, folder, fps=20):
        import cv2
        self.cv2 = cv2
        self.collector = collector
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.writer = cv2.VideoWriter(str(self.folder / "collection_process.mp4"),
                                     cv2.VideoWriter_fourcc(*"mp4v"), fps, (1280, 580))
        if not self.writer.isOpened():
            raise RuntimeError("Cannot create the scene video")
        self.renderer = None
        self.model = self.data = None
        self.phase = "Opening gripper"
        self.episode = -1
        self.cfg = {}
        self.last_sim = None
        self.last_wall = None
        self.native_viewer = None
        self.trace = []
        self.last_frame = None

    def bind(self, model, data, episode, cfg):
        if self.renderer is not None and model is not self.model:
            self.renderer.close()
            self.renderer = None
        self.model, self.data, self.episode, self.cfg = model, data, episode, cfg
        self.last_sim = self.last_wall = None
        self.phase = "Opening gripper / setting object pose"

    def sync(self, viewer):
        if self.model is None:
            return
        mujoco = self.collector.mujoco
        if viewer is not None and viewer is not self.native_viewer:
            with viewer.lock():
                viewer.cam.lookat[:] = [0.30, 0.0, 0.90]
                viewer.cam.distance = 1.25
                viewer.cam.azimuth = 135
                viewer.cam.elevation = -25
            self.native_viewer = viewer
        now = float(self.data.time)
        if self.last_sim is not None and now >= self.last_sim and now - self.last_sim < 1 / self.fps - 1e-8:
            return
        # Pace only fast simulation sections. Rendering can take longer than real time.
        if self.last_wall is not None and now >= self.last_sim:
            delay = min(now - self.last_sim, 1 / self.fps) - (time.perf_counter() - self.last_wall)
            if delay > 0:
                time.sleep(delay)
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        self.renderer.update_scene(self.data, camera="main_cam")
        overview = self.renderer.render().copy()
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.collector.OBJECT_BODY)
        camera.lookat[:] = self.data.xpos[body]
        camera.distance, camera.azimuth, camera.elevation = 0.30, 135, -25
        self.renderer.update_scene(self.data, camera=camera)
        detail = self.renderer.render().copy()
        frame = np.zeros((580, 1280, 3), dtype=np.uint8)
        frame[68:548] = self.cv2.cvtColor(np.concatenate([overview, detail], axis=1), self.cv2.COLOR_RGB2BGR)
        offset = np.asarray(self.cfg.get("object_position_offset_world_m", [0, 0, 0])) * 1000
        contact = np.asarray(self.cfg.get("grasp_offset_object_m", [0, 0, 0])) * 1000
        lines = [
            f"REAL SIMULATION | episode {self.episode:04d} | sim time {now:.2f} s | {self.phase}",
            f"World offset mm: {offset.tolist()} | RPY deg: {self.cfg.get('object_rpy_deg')} | Contact offset mm: {contact.tolist()}",
        ]
        for index, line in enumerate(lines):
            self.cv2.putText(frame, line, (12, 25 + index * 26), self.cv2.FONT_HERSHEY_SIMPLEX,
                             0.53, (235, 235, 235), 1, self.cv2.LINE_AA)
        self.cv2.putText(frame, "Robot overview                               Object / gripper close-up     |     Virtual fixture holds object",
                         (12, 570), self.cv2.FONT_HERSHEY_SIMPLEX, 0.57, (220, 220, 220), 1, self.cv2.LINE_AA)
        self.writer.write(frame)
        self.trace.append((self.episode, now, self.data.qpos.copy(), self.data.qvel.copy(), self.phase))
        if self.last_sim is None:
            self.save_png(self.folder / f"scene_episode_{self.episode:04d}_start.png", frame)
        self.last_frame = frame
        self.last_sim, self.last_wall = now, time.perf_counter()

    def save_png(self, path, frame):
        ok, encoded = self.cv2.imencode(".png", frame)
        if not ok:
            raise RuntimeError("Scene PNG encoding failed")
        Path(path).write_bytes(encoded.tobytes())

    def close(self):
        self.writer.release()
        if self.renderer is not None:
            self.renderer.close()
        if self.trace:
            np.savez_compressed(
                self.folder / "process_trace.npz",
                episode_index=np.array([item[0] for item in self.trace]),
                simulation_time_s=np.array([item[1] for item in self.trace]),
                qpos=np.stack([item[2] for item in self.trace]),
                qvel=np.stack([item[3] for item in self.trace]),
                phase=np.array([item[4] for item in self.trace]),
                video_fps=np.array(self.fps),
            )
        (self.folder / "process_video.json").write_text(json.dumps({
            "fps": self.fps, "frame_count": len(self.trace),
            "duration_s": len(self.trace) / self.fps,
            "description": "Actual simulator states sampled in simulation time; initialization and file-save delays are omitted.",
            "live_viewer": bool(self.native_viewer is not None),
        }, ensure_ascii=False, indent=2), encoding="utf-8")


def install_visual_main(collector, recorder):
    original_main = collector.main

    def visual_main():
        # This executes after run_prepared_object has installed the MCB adapters.
        original_sync, original_run = collector.sync_viewer, collector.run_episode
        original_joint = collector.move_arm_joint_space
        original_cartesian = collector.move_grasp_center_cartesian
        original_contact = collector.collect_contact_sequence
        cartesian_count = 0
        inside_cartesian = False

        def sync(viewer):
            original_sync(viewer)
            recorder.sync(viewer)

        def episode(*args, **kwargs):
            nonlocal cartesian_count
            cartesian_count = 0
            recorder.bind(kwargs["model"], kwargs["data"], kwargs["episode_idx"], kwargs["pose_cfg"])
            result = original_run(*args, **kwargs)
            recorder.sync(kwargs.get("viewer"))
            if recorder.last_frame is not None:
                recorder.save_png(recorder.folder / f"scene_episode_{recorder.episode:04d}_end.png", recorder.last_frame)
            return result

        def joint(*args, **kwargs):
            if not inside_cartesian:
                recorder.phase = "Moving arm to safe pose"
            return original_joint(*args, **kwargs)

        def cartesian(*args, **kwargs):
            nonlocal cartesian_count, inside_cartesian
            cartesian_count += 1
            recorder.phase = "Approach: safe to pregrasp" if cartesian_count == 1 else "Approach: pregrasp to object"
            inside_cartesian = True
            try:
                return original_cartesian(*args, **kwargs)
            finally:
                inside_cartesian = False

        def contact(*args, **kwargs):
            recorder.phase = "Closing gripper / capturing GelSight"
            return original_contact(*args, **kwargs)

        collector.sync_viewer, collector.run_episode = sync, episode
        collector.move_arm_joint_space, collector.move_grasp_center_cartesian = joint, cartesian
        collector.collect_contact_sequence = contact
        try:
            return original_main()
        finally:
            collector.sync_viewer, collector.run_episode = original_sync, original_run
            collector.move_arm_joint_space = original_joint
            collector.move_grasp_center_cartesian = original_cartesian
            collector.collect_contact_sequence = original_contact

    collector.main = visual_main


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obj", type=Path, default=core.models.DATASET_ROOT / "train" /
                        "Articulations, eyelets and other articulated joints" / "00011996.obj")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-configs", type=int, default=2)
    parser.add_argument("--sensor-clearance-mm", type=float, default=1.0)
    parser.add_argument("--headless", action="store_true", help="Record the video without a native viewer")
    args = parser.parse_args(argv)
    output = args.output or core.DEFAULT_OUTPUT / datetime.now().strftime("%Y%m%d_%H%M%S_visual_test")
    argv_core = ["--obj", str(args.obj), "--output", str(output), "--max-configs", str(args.max_configs),
                 "--sensor-clearance-mm", str(args.sensor_clearance_mm), "--run"]
    if args.headless:
        argv_core.append("--headless")
    options = core.parse_args(argv_core)
    prepared = core.prepare_object(options, args.obj)
    core.print_object_plan(options, prepared, output)
    collector = core._load_collecting_module()
    core.save_plan(output, prepared["plan"], False)
    recorder = SceneRecorder(collector, output / "preview")
    install_visual_main(collector, recorder)
    try:
        result = core.run_prepared_object(options, prepared, output, collector=collector)
    finally:
        recorder.close()
    (output / "preview" / "trial_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[VISUAL TEST RESULT] {json.dumps(result)}", flush=True)
    print(f"[VISUAL TEST OUTPUT] {output.resolve()}", flush=True)
    return 0 if result["valid"] and not result["incomplete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
