"""Checks geometry invariance, displacement units, timing, and cache provenance."""
import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tactile.features import (atomic_save_npz, cache_is_valid, farneback_flow,
                             feature_path, frame_intervals, source_fingerprint,
                             validate_cache_source)
from tactile.geometry import GripperGeometry, edge_attributes


class FeatureTests(unittest.TestCase):
    def test_intervals_are_actual_elapsed_time(self):
        result = frame_intervals([{"time": "1.0"}, {"time": "1.02"}, {"time": "1.08"}], 3)
        np.testing.assert_allclose(result, [0.02, 0.06])
        with self.assertRaises(ValueError):
            frame_intervals([{"time": "1.0"}, {"time": "1.0"}])
        with self.assertRaises(ValueError):
            frame_intervals([{"time": "1.0"}, {"time": "1.02"}], 3)

    @unittest.skipUnless(importlib.util.find_spec("cv2"), "OpenCV is optional for training")
    def test_known_translation_uses_normalized_displacement(self):
        import cv2
        rng = np.random.default_rng(8)
        source = rng.integers(0, 256, (128, 96), dtype=np.uint8)
        source = cv2.GaussianBlur(source, (5, 5), 0.7)
        moved = cv2.warpAffine(source, np.float32([[1, 0, 3], [0, 1, -2]]), (96, 128), borderMode=cv2.BORDER_REFLECT)
        rgb = np.repeat(np.stack((source, moved))[..., None], 3, axis=-1)
        flow = farneback_flow(np.stack((rgb, rgb, rgb)))
        self.assertEqual(flow.shape, (3, 1, 2, 128, 96))
        np.testing.assert_allclose(np.median(flow[0, 0, :, 20:-20, 20:-20], axis=(1, 2)), [3 / 96, -2 / 128], atol=0.003)

    def test_stale_and_foreign_sources_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            episode = Path(temporary) / "episode"
            episode.mkdir()
            rgb, signals = episode / "tactile_rgb.npz", episode / "signals.csv"
            rgb.write_bytes(b"sample")
            signals.write_text("time\n1\n2\n", encoding="utf-8")
            fingerprint = source_fingerprint([rgb, signals], {"method": "test"})
            cache = Path(temporary) / "flow.npz"
            atomic_save_npz(cache, {"schema_version": 1, "kind": "flow", "source_fingerprint": fingerprint}, flow=np.zeros(1))
            self.assertTrue(cache_is_valid(cache, fingerprint))
            validate_cache_source(cache, episode, "flow")
            with self.assertRaises(ValueError):
                validate_cache_source(cache, Path(temporary) / "other", "flow")
            signals.write_text("time\n1\n2\n3\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_cache_source(cache, episode, "flow")

    def test_episode_id_cannot_escape_cache_root(self):
        self.assertEqual(feature_path("cache/flow", "flow", "run/episode"), Path("cache/flow/run/episode/flow.npz"))
        for episode_id in ("../outside", "/absolute", "C:/absolute", "foo\\bar"):
            with self.assertRaises(ValueError):
                feature_path("cache", "flow", episode_id)

    @unittest.skipUnless(importlib.util.find_spec("mujoco"), "MuJoCo required for geometry reconstruction")
    def test_geometry_invariant_to_global_rigid_transform(self):
        import mujoco
        xml = '''<mujoco><worldbody><body name="global"><freejoint/>
        <geom type="sphere" size=".01" mass="1"/><body name="gripper_mount" pos=".1 .2 .3">
        <body name="f1" pos=".02 0 0"><joint name="finger1_hinge" axis="0 1 0"/>
          <geom type="sphere" size=".005" mass=".1"/><site name="finger1_gelsight_site" pos="0 0 .04"/></body>
        <body name="f2" pos="-.02 0 0"><joint name="finger2_hinge" axis="0 1 0"/>
          <geom type="sphere" size=".005" mass=".1"/><site name="finger2_gelsight_site" pos="0 0 .04"/></body>
        <body name="f3" pos="0 .02 0"><joint name="finger3_hinge" axis="1 0 0"/>
          <geom type="sphere" size=".005" mass=".1"/><site name="finger3_gelsight_site" pos="0 0 .04"/></body>
        </body></body></worldbody></mujoco>'''
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "robot.xml"
            path.write_text(xml, encoding="utf-8")
            geometry = GripperGeometry(path)
            pose, edges = geometry.replay(np.array([[.1, -.2, .3], [.2, -.1, .3]]))
            baseline = pose[-1]
            geometry.data.qpos[:7] = [1.0, 2.0, -0.5, np.cos(.4), 0, np.sin(.4), 0]
            mujoco.mj_forward(geometry.model, geometry.data)
            np.testing.assert_allclose(geometry.current_sensor_pose(), baseline, atol=1e-6)
            np.testing.assert_allclose(edges[:, :, :, :3], -edges[:, :, :, :3].swapaxes(1, 2), atol=1e-6)
            np.testing.assert_allclose(edges[:, :, :, 3], np.linalg.norm(edges[:, :, :, :3], axis=-1))
            np.testing.assert_allclose(np.diagonal(edges[..., 4], axis1=1, axis2=2), 1, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
