"""Synthetic samples only: format layouts, resume integrity, and safe cleanup."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import collect_mcb_gelsight_dataset as adapter
import collection_storage as storage


class CollectionStorageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = adapter.load_collector(adapter.DEFAULT_COLLECTOR, skip_taxim=True)
        cls.base.SAVE_GELSIGHT_IMAGES = True
        cls.base.SAVE_OPTICAL_FLOW = True

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="collection_formats_")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve() / "中文路径"
        self.root.mkdir()
        self.base.RUN_OUTPUT_ROOT = self.root
        self.episode = self.root / "物体" / "episode_0000"
        self.config = {"stage": "synthetic", "offset": [0.01, 0, 0]}

    def _samples(self):
        rng = np.random.default_rng(78)
        texture = rng.integers(0, 256, (36, 40, 3), dtype=np.uint8)
        # Distinct RGB channels detect an accidental RGB/BGR reversal in PNG I/O.
        texture[:, :, 0] //= 2
        samples = []
        for index in range(3):
            samples.append({
                "time": 0.1 * index,
                "phase": "hold",
                "gripper_ctrl": 0.2,
                "finger_q": [0.1] * 3,
                "contacts": [True] * 3,
                "contact_counts": [1] * 3,
                "normal_forces": [0.5] * 3,
                "grasp_center": [0, 0, 0],
                "object_pos": [0.1, 0.2, 0.3],
                "gel_centers_object": [[0, 0, 0]] * 3,
                "tool_pos": [0, 0, 0],
                "tool_R": np.eye(3).reshape(-1),
                "object_R": np.eye(3).reshape(-1),
                "gel_rotations_object": [np.eye(3).reshape(-1)] * 3,
                "images": {finger: np.roll(texture, index, axis=1) for finger in storage.FINGERS},
                "depth_images": {finger: np.full((36, 40), index / 8, np.float32)
                                 for finger in storage.FINGERS},
            })
        return samples

    def _metadata(self, paths):
        return json.loads(paths["metadata"].read_text(encoding="utf-8"))

    def test_all_layouts_lossless_formats_and_read_only_resume(self):
        for layout in storage.LAYOUTS:
            with self.subTest(layout=layout):
                self.base.OUTPUT_LAYOUT = layout
                self.base.RUN_OUTPUT_ROOT = self.root / layout
                episode = self.base.RUN_OUTPUT_ROOT / "物体" / "episode_0000"
                paths = storage.episode_paths(episode, self.base.RUN_OUTPUT_ROOT, layout)
                samples = self._samples()
                self.base.save_episode(episode, samples, {"valid": True, "pose_config": self.config})
                metadata = self._metadata(paths)
                self.assertTrue(metadata["collection_complete"])
                self.assertEqual(metadata["storage"]["layout"], layout)
                relative = json.loads(metadata["storage"]["run_relative_paths"])
                self.assertEqual(relative["signals"], paths["signals"].relative_to(self.base.RUN_OUTPUT_ROOT).as_posix())
                self.assertTrue(paths["signals"].is_file())
                self.assertTrue(paths["depth"].is_file())
                self.assertTrue(paths["flow"].is_file())
                if layout == "legacy":
                    with np.load(paths["rgb_npz"], allow_pickle=False) as rgb:
                        images = {finger: rgb[finger] for finger in storage.FINGERS}
                else:
                    images = storage.load_png_sequences(paths["rgb"], 3)
                    self.assertEqual(len(list(paths["rgb"].rglob("*.png"))), 9)
                    self.assertFalse(paths["rgb_npz"].exists())
                    self.assertFalse((episode / "tactile_rgb.npz").exists())
                    if layout == "by_type":
                        self.assertFalse(episode.exists())
                for finger in storage.FINGERS:
                    np.testing.assert_array_equal(images[finger], np.stack([s["images"][finger] for s in samples]))
                with np.load(paths["depth"], allow_pickle=False) as depth:
                    np.testing.assert_array_equal(depth["finger1"], np.stack([s["depth_images"]["finger1"] for s in samples]))
                with np.load(paths["flow"], allow_pickle=False) as flow:
                    self.assertEqual(flow["finger1"].shape, (2, 36, 40, 2))
                    self.assertEqual(flow["finger1"].dtype, np.float32)
                original = {p: p.read_bytes() for p in self.base.RUN_OUTPUT_ROOT.rglob("*") if p.is_file()}
                self.assertTrue(self.base.episode_output_complete(episode, metadata, self.config))
                self.assertEqual(original, {p: p.read_bytes() for p in original})
                self.assertFalse(self.base.episode_output_complete(episode, metadata, {"stage": "changed"}))

    def test_paths_are_grouped_by_type_or_by_episode(self):
        by_type = storage.episode_paths(self.episode, self.root, "by_type")
        self.assertEqual(by_type["rgb"], self.root / "rgb_png" / "物体" / "episode_0000")
        self.assertEqual(by_type["depth"], self.root / "depth_npz" / "物体" / "episode_0000" / "tactile_depth.npz")
        by_episode = storage.episode_paths(self.episode, self.root, "by_episode")
        self.assertEqual(by_episode["rgb"], self.episode / "rgb_png")
        self.assertEqual(by_episode["signals"], self.episode / "signals_csv" / "signals.csv")

    def test_cleanup_never_touches_another_sample(self):
        for layout in storage.LAYOUTS:
            with self.subTest(layout=layout):
                self.base.OUTPUT_LAYOUT = layout
                other = self.root / "物体" / "episode_0001"
                for episode in (self.episode, other):
                    self.base.save_episode(episode, self._samples(), {"valid": True, "pose_config": self.config})
                other_paths = storage.episode_paths(other, self.root, layout)
                protected = {p: p.read_bytes() for path in other_paths.values()
                             for p in ([path] if path.is_file() else path.rglob("*")) if p.is_file()}
                self.base.clear_episode_outputs(self.episode)
                paths = storage.episode_paths(self.episode, self.root, layout)
                for key in ("rgb", "depth", "flow", "signals"):
                    self.assertFalse(paths[key].exists(), key)
                self.assertTrue(paths["metadata"].is_file())
                self.assertEqual(protected, {p: p.read_bytes() for p in protected})

    def test_resume_detects_missing_rgb_and_failed_save_remains_incomplete(self):
        self.base.OUTPUT_LAYOUT = "by_type"
        paths = storage.episode_paths(self.episode, self.root, "by_type")
        self.base.save_episode(self.episode, self._samples(), {"valid": True, "pose_config": self.config})
        metadata = self._metadata(paths)
        (paths["rgb"] / "finger2" / "frame_0001.png").unlink()
        self.assertFalse(self.base.episode_output_complete(self.episode, metadata, self.config))
        with patch.object(self.base, "save_png_sequences", side_effect=OSError("simulated disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                self.base.save_episode(self.episode, self._samples(), {"valid": True, "pose_config": self.config})
        self.assertFalse(self._metadata(paths)["collection_complete"])
        self.assertFalse(paths["flow"].exists())
        self.assertFalse(self.base.episode_output_complete(self.episode, self._metadata(paths), self.config))

    def test_invalid_zero_frame_and_storage_mismatch(self):
        self.base.OUTPUT_LAYOUT = "by_type"
        paths = storage.episode_paths(self.episode, self.root, "by_type")
        self.base.save_episode(self.episode, [], {"valid": False, "pose_config": self.config})
        metadata = self._metadata(paths)
        self.assertTrue(self.base.episode_output_complete(self.episode, metadata, self.config))
        metadata["storage"]["layout"] = "by_episode"
        self.assertFalse(self.base.episode_output_complete(self.episode, metadata, self.config))

    def test_unsafe_paths_are_rejected_before_cleanup(self):
        for layout in storage.LAYOUTS:
            for episode in (self.root, self.root.parent / "outside", self.root / ".." / "outside"):
                with self.subTest(layout=layout, path=episode):
                    with self.assertRaises(ValueError):
                        storage.clear_episode_files(episode, self.root, layout)
        with self.assertRaises(ValueError):
            storage.episode_paths(self.episode, self.root, "unknown")


if __name__ == "__main__":
    unittest.main()
