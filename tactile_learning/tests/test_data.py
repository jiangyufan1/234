"""Tests cover leakage, malformed input, masks and stale temporal feature caches."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from tactile.data.index import FINGERS, build_index, read_json, read_signals, write_json
from tactile.data.splits import build_splits, validate_split
from tactile.data.dataset import TactileEpisodeDataset
from tactile.features import atomic_save_npz, source_fingerprint


def write_episode(path, label, condition, repeat=0):
    path.mkdir(parents=True)
    angle = np.deg2rad(condition * 20) / 2
    q = [float(np.cos(angle)), 0., 0., float(np.sin(angle))]
    if repeat:
        q = [-v for v in q]
    metadata = {"object": label, "valid": True, "stable_three_point_contact": True,
                "tactile_images_saved": True, "num_frames": 6,
                "pose_config": {"object_quat_wxyz": q, "grasp_offset_object_m": [condition * .020, 0., 0.],
                                "approach_azimuth_deg": 0., "approach_tilt_deg": 0., "gripper_roll_deg": 0.}}
    write_json(path / "metadata.json", metadata)
    image = np.full((6, 12, 8, 3), 90 + condition, dtype=np.uint8)
    np.savez_compressed(path / "tactile_rgb.npz", **{finger: image for finger in FINGERS})
    fields = ["frame", "time", "phase", "q_f1", "q_f2", "q_f3", "contact_f1", "contact_f2", "contact_f3"]
    with (path / "signals.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for frame in range(6):
            row = {"frame": frame, "time": 1. + frame * .02,
                   "phase": "precontact" if frame == 0 else "hold" if frame >= 4 else "dynamic_contact"}
            row.update({f"q_f{i}": .1 for i in range(1, 4)})
            row.update({f"contact_f{i}": int(frame > 0) for i in range(1, 4)})
            writer.writerow(row)


class DataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.data = self.base / "data"
        for label in ["can", "box"]:
            for condition in range(6):
                for repeat in range(2):
                    write_episode(self.data / label / f"episode_{condition}_{repeat}", label, condition, repeat)
        self.index_path, self.split_path = self.base / "index.json", self.base / "split.json"
        self.index = build_index(self.data, self.index_path)
        self.split = build_splits(self.index_path, self.split_path)
        self.config = {"height": 16, "width": 12, "static_frames": 4, "max_flow_frames": 7,
                       "use_flow": False, "use_geometry": False, "contact_source": "signals"}

    def tearDown(self):
        self.tmp.cleanup()

    def test_index_classes_and_legacy_complete_flag(self):
        self.assertEqual(self.index["classes"], ["box", "can"])
        self.assertEqual(len(self.index["episodes"]), 24)
        self.assertFalse(self.index["rejected"])

    def test_rgb_frame_mismatch_rejected(self):
        episode = Path(self.index["episodes"][0]["path"])
        np.savez_compressed(episode / "tactile_rgb.npz", **{f: np.zeros((5, 12, 8, 3), np.uint8) for f in FINGERS})
        result = build_index(self.data, self.index_path)
        self.assertEqual(len(result["rejected"]), 1)
        self.assertIn("expected", result["rejected"][0]["reason"])

    def test_non_monotonic_time_rejected(self):
        path = Path(self.index["episodes"][0]["path"]) / "signals.csv"
        with path.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        rows[1]["time"] = rows[0]["time"]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        with self.assertRaisesRegex(ValueError, "strictly increase"):
            read_signals(path)

    def test_all_protocols_keep_repeats_and_classes_together(self):
        for protocol in ["grouped", "position", "pose"]:
            split = build_splits(self.index_path, self.split_path, protocol=protocol, block_mm=15)
            assignments = {eid: part for part in ["train", "val", "test"] for eid in split[part]}
            for label in self.index["classes"]:
                for condition in range(6):
                    self.assertEqual(assignments[f"{label}/episode_{condition}_0"], assignments[f"{label}/episode_{condition}_1"])
                for part in ["train", "val", "test"]:
                    self.assertTrue(any(eid.startswith(label + "/") for eid in split[part]))

    def test_too_few_groups_fails_without_episode_fallback(self):
        with self.assertRaisesRegex(ValueError, "at least 3"):
            build_splits(self.index_path, self.split_path, protocol="position", block_mm=1000)

    def test_large_position_buffer_reports_empty_partition(self):
        with self.assertRaisesRegex(ValueError, "empty partition"):
            build_splits(self.index_path, self.split_path, protocol="position", buffer_mm=200)

    def test_hand_edited_split_leakage_rejected(self):
        split = json.loads(json.dumps(self.split))
        moved = split["train"].pop()
        split["test"].append(moved)
        with self.assertRaisesRegex(ValueError, "Group leakage"):
            validate_split(self.index, split)

    def test_static_padding_and_masks(self):
        dataset = TactileEpisodeDataset(self.index_path, self.split_path, "train", self.config)
        sample = dataset[0]
        self.assertEqual(tuple(sample["rgb"].shape), (3, 4, 3, 16, 12))
        self.assertEqual(sample["rgb_mask"].sum().item(), 12)
        self.assertTrue(sample["node_mask"].all())

        # RGB samples include dynamic_contact and hold.
        expected_rgb_times = torch.tensor([1.02, 1.04, 1.08, 1.10])
        self.assertTrue(torch.allclose(
            sample["rgb_times"], expected_rgb_times, atol=1e-6
        ))
        self.assertTrue(sample["rgb_mask"].all())
        self.assertFalse(sample["flow_mask"].any())
        self.assertEqual(dataset.labels[0], sample["label"].item())

    def test_source_mutation_requires_reindex(self):
        dataset = TactileEpisodeDataset(self.index_path, self.split_path, "train", self.config)
        path = Path(dataset.episodes[0]["path"]) / "metadata.json"
        path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Source changed"):
            dataset[0]

    def _write_flow(self, dataset, *, wrong_dt=False):
        record = dataset.episodes[0]
        path = Path(record["path"])
        config = {"backend": "farneback", "height": 12, "width": 8}
        metadata = {"schema_version": 1, "kind": "flow", "config": config,
                    "backend": "farneback",
                    "source_fingerprint": source_fingerprint([path / "signals.csv", path / "tactile_rgb.npz"], config),
                    "units": "u/width,v/height per adjacent pair", "finger_order": list(FINGERS)}
        atomic_save_npz(self.base / "flow" / record["id"] / "flow.npz", metadata,
                        flow=np.full((3, 5, 2, 12, 8), .2, np.float32),
                        dt=np.full(5, .03 if wrong_dt else .02, np.float32))

    def test_flow_normalized_units_and_padding(self):
        config = {**self.config, "use_flow": True, "flow_cache": str(self.base / "flow")}
        dataset = TactileEpisodeDataset(self.index_path, self.split_path, "train", config)
        self._write_flow(dataset)
        sample = dataset[0]
        self.assertEqual(sample["flow_mask"].sum().item(), 15)
        self.assertTrue(torch.allclose(sample["flow"][:, :5], torch.tensor(.2)))
        self.assertTrue(torch.all(sample["flow"][:, 5:] == 0))
        self.assertTrue(torch.allclose(sample["flow_dt"][:5], torch.tensor(.02)))

        expected_start = torch.tensor([1.00, 1.02, 1.04, 1.06, 1.08])
        expected_end = torch.tensor([1.02, 1.04, 1.06, 1.08, 1.10])

        self.assertTrue(torch.allclose(sample["flow_start"][:5], expected_start, atol=1e-6))
        self.assertTrue(torch.allclose(sample["flow_end"][:5], expected_end, atol=1e-6))
        self.assertTrue(torch.allclose(
            sample["flow_end"][:5] - sample["flow_start"][:5],
            sample["flow_dt"][:5],
            atol=1e-6,
        ))

    def test_required_flow_missing_or_wrong_time_fails(self):
        config = {**self.config, "use_flow": True, "flow_cache": str(self.base / "flow")}
        dataset = TactileEpisodeDataset(self.index_path, self.split_path, "train", config)
        with self.assertRaises(ValueError):
            dataset[0]
        self._write_flow(dataset, wrong_dt=True)
        with self.assertRaisesRegex(ValueError, "dt does not match"):
            dataset[0]

    def test_mixed_flow_backend_rejected(self):
        config = {**self.config, "use_flow": True, "flow_cache": str(self.base / "flow"), "flow_backend": "raft"}
        dataset = TactileEpisodeDataset(self.index_path, self.split_path, "train", config)
        self._write_flow(dataset)
        with self.assertRaisesRegex(ValueError, "backend differs"):
            dataset[0]


if __name__ == "__main__":
    unittest.main()
