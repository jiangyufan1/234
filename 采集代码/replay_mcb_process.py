#!/usr/bin/env python
"""Render a saved state trace without advancing or modifying the simulation.

The right panel hides all geometry except the original object surface and the
three gel pads. Distances come from recomputed raw MuJoCo contacts at each saved
qpos; a positive contact distance is still a gap (the contact margin can create
constraints before surfaces overlap). No grasp success is inferred here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from tactile_optical_flow import _require_cv2


def load_model(path: Path):
    """Use XML text so Windows paths containing Chinese characters work."""
    root = ET.fromstring(path.read_text(encoding="utf-8"))
    compiler = root.find("compiler")
    if compiler is not None:
        for key in ("meshdir", "texturedir"):
            value = compiler.get(key)
            if value and not Path(value).is_absolute():
                compiler.set(key, str((path.parent / value).resolve()))
    for include in root.iter("include"):
        value = include.get("file")
        if value and not Path(value).is_absolute():
            include.set("file", str((path.parent / value).resolve()))
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def save_png(cv2, path, frame):
    ok, encoded = cv2.imencode(".png", frame)
    if not ok:
        raise RuntimeError(f"Cannot encode {path}")
    path.write_bytes(encoded.tobytes())


def episode_metadata(run, episode):
    matches = list((run / "metadata_json").glob(f"*/episode_{episode:04d}/metadata.json"))
    if not matches:
        matches = list(run.glob(f"*/episode_{episode:04d}/metadata.json"))
    if len(matches) == 1:
        return json.loads(matches[0].read_text(encoding="utf-8"))
    return {}


def render_replay(run: Path, model_path: Path | None = None):
    cv2 = _require_cv2()
    preview = run / "preview"
    trace_path = preview / "process_trace.npz"
    with np.load(trace_path, allow_pickle=False) as source:
        trace = {key: source[key].copy() for key in source.files}
    qpos, qvel = trace["qpos"], trace["qvel"]
    episodes, times, phases = (trace[key] for key in
                              ("episode_index", "simulation_time_s", "phase"))
    fps = float(trace["video_fps"].item())
    if len(qpos) == 0 or fps <= 0 or not np.isfinite(qpos).all():
        raise ValueError("Empty or invalid trace")
    if model_path is None:
        models = list((run / "models").glob("*.xml"))
        if len(models) != 1:
            raise ValueError("Specify --model when the run contains more than one model")
        model_path = models[0]
    model = load_model(model_path)
    if qpos.shape != (len(times), model.nq) or qvel.shape != (len(times), model.nv):
        raise ValueError("Trace dimensions do not match the model")
    data = mujoco.MjData(model)
    gel_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"finger{i}_gel")
               for i in (1, 2, 3)]
    object_ids = [i for i in range(model.ngeom)
                  if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith("mcb_part_")]
    visual_ids = [i for i in object_ids
                  if "visual" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "")]
    if min(gel_ids) < 0 or not object_ids:
        raise ValueError("Expected three finger gel geoms and MCB object geoms")
    if not visual_ids:
        visual_ids = object_ids[:1]
    object_body = int(model.geom_bodyid[visual_ids[0]])
    allowed = set(gel_ids + visual_ids)
    colors = {gel_ids[0]: (1, .15, .15, .75), gel_ids[1]: (.1, 1, .2, .75),
              gel_ids[2]: (1, .85, .1, .75)}
    colors.update({i: (.10, .60, 1, 1) for i in visual_ids})
    option = mujoco.MjvOption()
    option.geomgroup[:] = 1
    option.sitegroup[:] = 0
    option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False
    option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance, camera.azimuth, camera.elevation = .22, 135, -22
    video = preview / "collection_process_cutaway.mp4"
    width, height = 1280, 620
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("Cannot create replay video")
    renderer = mujoco.Renderer(model, height=480, width=640)
    metadata = {int(e): episode_metadata(run, int(e)) for e in np.unique(episodes)}
    diagnostics = []
    last_frame = None
    try:
        for index in range(len(qpos)):
            data.qpos[:] = qpos[index]
            data.qvel[:] = qvel[index]
            data.time = float(times[index])
            mujoco.mj_forward(model, data)  # Computes kinematics/contacts; never mj_step.
            camera.lookat[:] = data.xpos[object_body]
            camera.distance = .22
            renderer.update_scene(data, camera=camera, scene_option=option)
            left = renderer.render().copy()
            camera.distance = .11
            renderer.update_scene(data, camera=camera, scene_option=option)
            for geom in renderer.scene.geoms[:renderer.scene.ngeom]:
                if geom.objtype != mujoco.mjtObj.mjOBJ_GEOM or geom.objid not in allowed:
                    geom.rgba[3] = 0
                else:
                    geom.rgba[:] = colors[int(geom.objid)]
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
            right = renderer.render().copy()
            distances = [None, None, None]
            counts = [0, 0, 0]
            for contact in data.contact:
                g1, g2 = int(contact.geom1), int(contact.geom2)
                for finger, gel in enumerate(gel_ids):
                    if ((g1 == gel and g2 in object_ids) or
                            (g2 == gel and g1 in object_ids)):
                        value = float(contact.dist) * 1000
                        distances[finger] = value if distances[finger] is None else min(distances[finger], value)
                        counts[finger] += 1
            episode = int(episodes[index])
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[84:564] = cv2.cvtColor(np.concatenate((left, right), axis=1), cv2.COLOR_RGB2BGR)
            verified = metadata[episode].get("grasp_verified") is True
            status = ("Recorded grasp_verified=true; this is a state replay" if verified else
                      "Fixture-held tactile trial; not a free-grasp verification")
            lines = [
                (f"SAVED STATE REPLAY | episode {episode:04d} | sim {times[index]:.2f} s | {phases[index]}", 23),
                (status, 48),
                ("Object: blue | Gel pads: F1 red, F2 green, F3 yellow | Geometry and recorded positions unchanged", 587),
            ]
            for text, y in lines:
                cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, .53,
                            (235, 235, 235), 1, cv2.LINE_AA)
            cv2.putText(frame, "Original gripper close-up", (12, 74), cv2.FONT_HERSHEY_SIMPLEX,
                        .53, (235, 235, 235), 1, cv2.LINE_AA)
            cv2.putText(frame, "Same state, 2x zoom: object and gel pads", (652, 74), cv2.FONT_HERSHEY_SIMPLEX,
                        .53, (235, 235, 235), 1, cv2.LINE_AA)
            distance_text = " | ".join(f"F{i + 1}: {v:+.3f} mm" if v is not None else
                                       f"F{i + 1}: no contact record" for i, v in enumerate(distances))
            cv2.putText(frame, "Raw min contact distance: " + distance_text + " | positive = gap",
                        (12, 611), cv2.FONT_HERSHEY_SIMPLEX, .48, (230, 230, 230), 1, cv2.LINE_AA)
            writer.write(frame)
            diagnostics.append({"frame": index, "episode": episode, "time_s": float(times[index]),
                                "raw_min_contact_distance_mm": distances, "raw_contact_counts": counts})
            last_frame = frame
            if index == len(qpos) - 1 or episodes[index + 1] != episode:
                save_png(cv2, preview / f"cutaway_episode_{episode:04d}_end.png", frame)
            if index % 100 == 0:
                print(f"[REPLAY] {index + 1}/{len(qpos)}", flush=True)
    finally:
        writer.release()
        renderer.close()
    save_png(cv2, preview / "cutaway_end.png", last_frame)
    capture = cv2.VideoCapture(str(video))
    decoded = 0
    while True:
        ok, image = capture.read()
        if not ok:
            break
        if image.shape != (height, width, 3):
            raise ValueError("Unexpected decoded video frame shape")
        decoded += 1
    capture.release()
    if decoded != len(qpos):
        raise ValueError(f"Expected {len(qpos)} replay frames, decoded {decoded}")
    report = {"trace": str(trace_path), "model": str(model_path), "frames": decoded,
              "fps": fps, "duration_s": decoded / fps, "physics_steps": 0,
              "geometry_changes": False, "state_source": "saved qpos and qvel; mj_forward only",
              "distance_note": "Raw MuJoCo contact.dist; positive means a surface gap. No record is not zero distance.",
              "frames_diagnostics": diagnostics}
    (preview / "cutaway_replay.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[REPLAY COMPLETE] {video} | {decoded} decoded frames", flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--model", type=Path)
    args = parser.parse_args(argv)
    render_replay(args.run.resolve(), args.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
