"""Small synthetic integration test, never an object-recognition benchmark.

Creates two artificial classes and disjoint repeated conditions in a temporary
directory, preprocesses continuous flow, trains, resumes and evaluates. With
--xml it also exercises actual MuJoCo encoder kinematics and the full graph.
"""
from __future__ import annotations

import argparse
import csv
import tempfile
from pathlib import Path

import numpy as np

from tactile.config import load_config, resolve_config
from tactile.data.index import build_index
from tactile.data.splits import build_splits
from tactile.engine import evaluate, train
from tactile.features import FEATURE_VERSION, compute_episode_flow
from tactile.runtime import write_json


def make_fixture(root):
    rng = np.random.default_rng(123)
    for label in range(2):
        for condition in range(3):
            for repeat in range(2):
                folder = root / f"synthetic_{label}" / f"episode_{condition}_{repeat}"
                folder.mkdir(parents=True)
                frames, height, width = 8, 32, 24
                base = rng.integers(20, 180, (height, width, 3), dtype=np.uint8)
                base[8:20, 5:16, label] = 240
                arrays = {f"finger{i}": np.stack([np.roll(base, t//2+i, axis=1) for t in range(frames)])
                          for i in range(1, 4)}
                np.savez_compressed(folder / "tactile_rgb.npz", **arrays)
                write_json(folder / "metadata.json", {
                    "object": f"synthetic_{label}", "valid": True, "reason": "synthetic_smoke_only",
                    "collection_complete": True, "stable_three_point_contact": True,
                    "tactile_images_saved": True, "num_frames": frames,
                    "pose_config": {"pose_group": f"synthetic_pose_{condition}",
                        "object_quat_wxyz": [float(np.cos(condition*.2)), 0, 0, float(np.sin(condition*.2))],
                        "grasp_offset_object_m": [condition*.02, 0, 0],
                        "approach_azimuth_deg": 0, "approach_tilt_deg": 0, "gripper_roll_deg": 0}})
                with (folder / "signals.csv").open("w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["frame", "time", "phase", "q_f1", "q_f2", "q_f3", "contact_f1", "contact_f2", "contact_f3"])
                    for t in range(frames):
                        phase = "precontact" if t == 0 else "dynamic_contact" if t < 5 else "hold"
                        writer.writerow([t, t*.02, phase, -.01*t, -.012*t, -.013*t, *([int(t>0)]*3)])


def run(folder, xml=None, device="cpu", workers=0):
    root = folder / "episodes"
    make_fixture(root)
    index_path, split_path = folder / "index.json", folder / "split.json"
    index = build_index(root, index_path)
    if index["rejected"]:
        raise AssertionError(index["rejected"])
    build_splits(index_path, split_path)
    flow_config = {"version": FEATURE_VERSION, "backend": "farneback", "height": 32, "width": 24,
                   "normalization": "u/width,v/height", "consecutive_pairs": True}
    geometry = None
    if xml:
        from tactile.geometry import GripperGeometry
        geometry = GripperGeometry(xml)
    for episode in index["episodes"]:
        compute_episode_flow(episode, folder / "flow", flow_config)
        if geometry:
            geometry.compute_episode(episode, folder / "geometry")
    variant = "full" if xml else "dynamic_concat"
    config = load_config(variant=variant)
    config["data"].update(height=32, width=24, static_frames=2, max_flow_frames=7,
                          flow_cache=str(folder/"flow"), geometry_cache=str(folder/"geometry"))
    config["model"].update(backbone="tiny", feature_dim=32)
    config["training"].update(epochs=1, batch_size=2, num_workers=workers, log_every=1)
    config = resolve_config(config)
    output = folder / "run"
    train(index_path, split_path, config, output, device, variant=variant)
    config["training"]["epochs"] = 2
    train(index_path, split_path, config, output, device, resume=output/"last.pt", variant=variant)
    metrics = evaluate(output/"best.pt", folder/"evaluation", device)
    assert metrics["count"] == 4
    print("PASS: synthetic index -> split -> features -> train -> resume -> test. No real-data accuracy claim.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--xml", help="Optional robot scene XML to exercise full geometry/graph model")
    p.add_argument("--device", default="cpu")
    p.add_argument("--workers", type=int, default=0, help="Use 2 to also exercise Windows DataLoader spawn")
    args = p.parse_args()
    with tempfile.TemporaryDirectory(prefix="tactile_smoke_") as folder:
        run(Path(folder), args.xml, args.device, args.workers)


if __name__ == "__main__":
    main()
