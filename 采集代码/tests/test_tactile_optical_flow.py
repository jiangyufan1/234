"""Numerical and synchronization checks; synthetic data only, no collection imports."""

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "tactile_optical_flow.py"
SPEC = importlib.util.spec_from_file_location("tactile_optical_flow", MODULE_PATH)
flow_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(flow_module)


class TactileOpticalFlowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.episode = Path(self.temporary.name) / "episode_0000"
        self.episode.mkdir()
        self.cv2 = flow_module._require_cv2()
        random = np.random.default_rng(318)
        texture = random.integers(0, 256, size=(96, 112), dtype=np.uint8)
        self.texture = self.cv2.GaussianBlur(texture, (3, 3), 0.6)

    def _sequence(self, dx, dy, count=3):
        frames = []
        for index in range(count):
            transform = np.float32([[1, 0, index * dx], [0, 1, index * dy]])
            gray = self.cv2.warpAffine(
                self.texture, transform, (112, 96), borderMode=self.cv2.BORDER_REFLECT
            )
            frames.append(np.repeat(gray[:, :, None], 3, axis=2))
        return np.stack(frames)

    def _write_signals(self, times=(0.0, 0.1, 0.35), frames=None):
        with (self.episode / "signals.csv").open("w", encoding="utf-8", newline="") as target:
            writer = csv.writer(target)
            writer.writerow(["frame", "time", "phase"])
            for index, time in enumerate(times):
                writer.writerow([index if frames is None else frames[index], time, "hold"])

    def _write_episode(self, times=(0.0, 0.1, 0.35), images=None):
        if images is None:
            images = {
                "finger1": self._sequence(3, 1),
                "finger2": self._sequence(-2, -3),
                "finger3": self._sequence(0, 0),
            }
        np.savez_compressed(self.episode / "tactile_rgb.npz", **images)
        self._write_signals(times)

    def _write_png_sequences(self, rgb_path, images=None):
        if images is None:
            images = {finger: self._sequence(index - 1, index) for index, finger in enumerate(flow_module.FINGERS)}
        for finger, frames in images.items():
            directory = rgb_path / finger
            directory.mkdir(parents=True, exist_ok=True)
            for index, frame in enumerate(frames):
                encoded, buffer = self.cv2.imencode(".png", self.cv2.cvtColor(frame, self.cv2.COLOR_RGB2BGR))
                self.assertTrue(encoded)
                (directory / f"frame_{index:04d}.png").write_bytes(buffer.tobytes())
        return images

    def _write_layout(self, run, layout, object_name="物体甲", episode_name="episode_0000", images=None):
        if layout == "by_type":
            relative_episode = Path(object_name) / episode_name
            rgb_path = run / "rgb_png" / relative_episode
            signals_path = run / "signals_csv" / relative_episode / "signals.csv"
            output_path = run / "flow_npz" / relative_episode / "tactile_flow.npz"
            episode = run / relative_episode
        else:
            episode = run / object_name / episode_name
            rgb_path = episode / "rgb_png"
            signals_path = episode / "signals_csv" / "signals.csv"
            output_path = episode / "flow_npz" / "tactile_flow.npz"
        self._write_png_sequences(rgb_path, images)
        signals_path.parent.mkdir(parents=True, exist_ok=True)
        signals_path.write_text("frame,time\n0,0\n1,0.1\n2,0.35\n", encoding="utf-8")
        return {"episode_dir": episode, "rgb_path": rgb_path, "signals_path": signals_path, "output_path": output_path}

    def test_known_translation_and_independent_fingers_at_original_resolution(self):
        self._write_episode()
        result = flow_module.save_episode_flow(self.episode)
        self.assertEqual(result["status"], "saved")
        self.assertEqual(result["frame_pairs"], 2)
        json.dumps(result)  # Return value can be placed directly in metadata.json.
        with np.load(self.episode / "tactile_flow.npz", allow_pickle=False) as archive:
            for finger, expected in {"finger1": [3, 1], "finger2": [-2, -3], "finger3": [0, 0]}.items():
                flow = archive[finger]
                self.assertEqual(flow.shape, (2, 96, 112, 2))
                self.assertEqual(flow.dtype, np.float32)
                for pair in flow:
                    median = np.median(pair[16:-16, 16:-16], axis=(0, 1))
                    np.testing.assert_allclose(median, expected, atol=0.25)
            # Identical displacements with unequal dt stay in pixels/frame_pair.
            np.testing.assert_array_equal(archive["frame_from"], [0, 1])
            np.testing.assert_array_equal(archive["frame_to"], [1, 2])
            np.testing.assert_allclose(archive["time_start"], [0.0, 0.1])
            np.testing.assert_allclose(archive["time_end"], [0.1, 0.35])
            np.testing.assert_allclose(archive["dt"], [0.1, 0.25])
            self.assertEqual(archive["units"].item(), "pixels/frame_pair")
            metadata = json.loads(archive["metadata"].item())
            self.assertEqual(metadata["direction"], "frame_from -> frame_to")
            self.assertEqual(metadata["source"]["rgb"]["file"], "tactile_rgb.npz")
            for name in archive.files:
                self.assertFalse(archive[name].dtype.hasobject, name)

    def test_time_count_nonmonotonic_and_nonfinite_rejected(self):
        self._write_episode()
        for times, error in [
            ((0, 0.1), "synchronization mismatch"),
            ((0, 0.1, 0.1), "strictly increasing"),
            ((0, 0.2, 0.1), "strictly increasing"),
            ((0, float("nan"), 0.2), "finite"),
        ]:
            with self.subTest(times=times):
                self._write_signals(times)
                with self.assertRaisesRegex(ValueError, error):
                    flow_module.save_episode_flow(self.episode)
                self.assertFalse((self.episode / "tactile_flow.npz").exists())

    def test_missing_frame_and_unsynchronized_finger_sequences_rejected(self):
        self._write_episode()
        self._write_signals(frames=[0, 2, 3])
        with self.assertRaisesRegex(ValueError, "contiguous"):
            flow_module.save_episode_flow(self.episode)
        for images in [
            {"finger1": self._sequence(1, 0), "finger2": self._sequence(1, 0)},
            {"finger1": self._sequence(1, 0), "finger2": self._sequence(1, 0, 2), "finger3": self._sequence(0, 1)},
            {finger: self._sequence(1, 0, 1) for finger in flow_module.FINGERS},
            {finger: self._sequence(1, 0).astype(np.float32) for finger in flow_module.FINGERS},
        ]:
            with self.subTest(shapes={key: value.shape for key, value in images.items()}):
                self._write_episode(images=images)
                with self.assertRaises(ValueError):
                    flow_module.save_episode_flow(self.episode)

    def test_cache_checks_source_content_and_explicit_overwrite(self):
        self._write_episode()
        flow_module.save_episode_flow(self.episode)
        flow_path = self.episode / "tactile_flow.npz"
        original = flow_path.read_bytes()
        self.assertEqual(flow_module.save_episode_flow(self.episode)["status"], "reused")
        self.assertEqual(flow_path.read_bytes(), original)
        self._write_signals((0, 0.2, 0.4))
        with self.assertRaisesRegex(FileExistsError, "stale or invalid"):
            flow_module.save_episode_flow(self.episode)
        self.assertEqual(flow_path.read_bytes(), original)
        self.assertEqual(flow_module.save_episode_flow(self.episode, overwrite=True)["status"], "saved")
        with np.load(flow_path, allow_pickle=False) as archive:
            np.testing.assert_allclose(archive["dt"], [0.2, 0.2])

    def test_contact_masks_follow_each_finger_and_each_endpoint(self):
        self._write_episode()
        (self.episode / "signals.csv").write_text(
            "frame,time,contact_count_f1,contact_count_f2,contact_count_f3\n"
            "0,0.0,1,0,0\n1,0.1,2,1,0\n2,0.35,0,1,0\n", encoding="utf-8"
        )
        metadata = flow_module.save_episode_flow(self.episode)
        self.assertTrue(metadata["contact_mask"]["available"])
        self.assertFalse(metadata["ground_truth"])
        with np.load(self.episode / "tactile_flow.npz", allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive["contact_start"], [[1, 0, 0], [1, 1, 0]])
            np.testing.assert_array_equal(archive["contact_end"], [[1, 1, 0], [0, 1, 0]])
            np.testing.assert_array_equal(archive["contact_pair"], [[1, 0, 0], [0, 1, 0]])
            self.assertEqual(archive["contact_pair"].dtype, np.bool_)
        self.assertEqual(flow_module.save_episode_flow(self.episode)["status"], "reused")

    def test_failed_atomic_write_keeps_existing_archive(self):
        self._write_episode()
        flow_module.save_episode_flow(self.episode)
        flow_path = self.episode / "tactile_flow.npz"
        original = flow_path.read_bytes()
        with patch.object(flow_module.np, "savez_compressed", side_effect=OSError("disk write failed")):
            with self.assertRaisesRegex(OSError, "disk write failed"):
                flow_module.save_episode_flow(self.episode, overwrite=True)
        self.assertEqual(flow_path.read_bytes(), original)
        self.assertEqual(list(self.episode.glob(".tactile_flow.*.tmp")), [])

    def test_missing_opencv_raises_clear_error(self):
        self._write_episode()
        with patch.dict(sys.modules, {"cv2": None}):
            with self.assertRaisesRegex(RuntimeError, "requires OpenCV"):
                flow_module.save_episode_flow(self.episode)

    def test_cli_scans_each_episode_separately(self):
        self._write_episode()
        second = self.episode.parent / "episode_0001"
        second.mkdir()
        # Deliberately reverse a second episode's motion; never pair the two episodes.
        np.savez_compressed(second / "tactile_rgb.npz", **{
            finger: self._sequence(-1, 2, 2) for finger in flow_module.FINGERS
        })
        (second / "signals.csv").write_text("frame,time\n0,9\n1,9.5\n", encoding="utf-8")
        self.assertEqual(flow_module.main(["--root", str(self.episode.parent)]), 0)
        for episode, expected_pairs in [(self.episode, 2), (second, 1)]:
            with np.load(episode / "tactile_flow.npz", allow_pickle=False) as archive:
                self.assertEqual(len(archive["finger1"]), expected_pairs)

    def test_png_and_npz_lossless_sequences_produce_identical_flow(self):
        # Unequal color channels ensure a mistaken BGR/RGB interpretation fails.
        images = {}
        for index, finger in enumerate(flow_module.FINGERS):
            gray = self._sequence(index - 1, index)
            images[finger] = np.stack((gray[..., 0], 255 - gray[..., 1], gray[..., 2] // 2), axis=-1)
        self._write_episode(images=images)
        flow_module.save_episode_flow(self.episode)
        paths = self._write_layout(self.episode.parent / "中文路径", "by_type", images=images)
        result = flow_module.save_episode_flow(**paths)
        self.assertEqual(result["status"], "saved")
        self.assertEqual(result["source"]["rgb"]["format"], "png_rgb")
        self.assertEqual(len(result["source"]["rgb"]["files"]), 9)
        for source in result["source"]["rgb"]["files"]:
            image_path = paths["rgb_path"] / source["file"]
            self.assertEqual(source["sha256"], hashlib.sha256(image_path.read_bytes()).hexdigest())
            self.assertEqual(source["bytes"], image_path.stat().st_size)
        with np.load(self.episode / "tactile_flow.npz", allow_pickle=False) as legacy, np.load(paths["output_path"], allow_pickle=False) as png:
            for name in (*flow_module.FINGERS, "frame_from", "frame_to", "time_start", "time_end", "dt"):
                np.testing.assert_array_equal(legacy[name], png[name])
        loaded = flow_module._load_rgb(paths["rgb_path"])
        for finger in flow_module.FINGERS:
            np.testing.assert_array_equal(loaded[finger], images[finger])

    def test_png_cache_fingerprints_each_file_and_signals(self):
        paths = self._write_layout(self.episode.parent / "分类保存", "by_type")
        flow_module.save_episode_flow(**paths)
        original = paths["output_path"].read_bytes()
        self.assertEqual(flow_module.save_episode_flow(**paths)["status"], "reused")
        image_path = paths["rgb_path"] / "finger2" / "frame_0001.png"
        image = self.cv2.imdecode(np.frombuffer(image_path.read_bytes(), dtype=np.uint8), self.cv2.IMREAD_COLOR)
        image[20, 20] = [1, 2, 3]
        _, encoded = self.cv2.imencode(".png", image)
        image_path.write_bytes(encoded.tobytes())
        with self.assertRaisesRegex(FileExistsError, "stale or invalid"):
            flow_module.save_episode_flow(**paths)
        self.assertEqual(paths["output_path"].read_bytes(), original)
        self.assertEqual(flow_module.save_episode_flow(**paths, overwrite=True)["status"], "saved")
        paths["signals_path"].write_text("frame,time\n0,0\n1,0.2\n2,0.4\n", encoding="utf-8")
        with self.assertRaisesRegex(FileExistsError, "stale or invalid"):
            flow_module.save_episode_flow(**paths)

    def test_png_missing_unsynchronized_duplicate_and_invalid_frames_rejected(self):
        for failure in ("gap", "different_count", "duplicate", "missing_finger", "corrupt", "empty", "different_shape", "bad_name"):
            with self.subTest(failure=failure):
                paths = self._write_layout(self.episode.parent / failure, "by_type")
                image_path = paths["rgb_path"] / "finger2" / "frame_0001.png"
                if failure == "gap":
                    image_path.rename(image_path.with_name("frame_0003.png"))
                elif failure == "different_count":
                    (image_path.parent / "frame_0002.png").unlink()
                elif failure == "duplicate":
                    image_path.with_name("frame_00001.png").write_bytes(image_path.read_bytes())
                elif failure == "missing_finger":
                    image_path.parent.rename(image_path.parent.with_name("wrong_finger"))
                elif failure == "corrupt":
                    image_path.write_bytes(b"invalid PNG data")
                elif failure == "empty":
                    image_path.write_bytes(b"")
                elif failure == "different_shape":
                    _, encoded = self.cv2.imencode(".png", np.zeros((20, 20, 3), np.uint8))
                    image_path.write_bytes(encoded.tobytes())
                else:
                    image_path.rename(image_path.with_name("frame.png"))
                with self.assertRaises(ValueError):
                    flow_module.save_episode_flow(**paths)
                self.assertFalse(paths["output_path"].exists())

    def test_cli_recognizes_both_png_layouts_and_keeps_episode_boundaries(self):
        root = self.episode.parent / "混合布局"
        paths = [
            self._write_layout(root / "episode_grouped", "by_type", episode_name="episode_0000"),
            self._write_layout(root / "episode_grouped", "by_type", episode_name="episode_0001", images={finger: self._sequence(-2, 1) for finger in flow_module.FINGERS}),
            self._write_layout(root / "分样本", "by_episode", episode_name="episode_0000"),
        ]
        sentinel = root / "metadata.json"
        sentinel.write_text('{"user_metadata": true}', encoding="utf-8")
        self.assertEqual(flow_module.main(["--root", str(root)]), 0)
        for job in paths:
            self.assertTrue(job["output_path"].is_file())
            self.assertFalse((job["episode_dir"] / "tactile_flow.npz").exists())
        with np.load(paths[1]["output_path"], allow_pickle=False) as archive:
            np.testing.assert_allclose(np.median(archive["finger1"][0, 16:-16, 16:-16], axis=(0, 1)), [-2, 1], atol=0.25)
        self.assertEqual(sentinel.read_text(encoding="utf-8"), '{"user_metadata": true}')
        self.assertEqual(len(flow_module._discover_inputs(paths[1]["rgb_path"])), 1)
        self.assertEqual(flow_module.main(["--root", str(paths[1]["rgb_path"])]), 0)

    def test_cli_prefers_png_when_same_episode_also_has_legacy_npz(self):
        paths = self._write_layout(self.episode.parent / "同一采集", "by_episode")
        np.savez_compressed(paths["episode_dir"] / "tactile_rgb.npz", **{
            finger: self._sequence(5, 5) for finger in flow_module.FINGERS
        })
        jobs = flow_module._discover_inputs(paths["episode_dir"])
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0]["rgb_path"], paths["rgb_path"].resolve())
        self.assertEqual(flow_module.main(["--root", str(paths["episode_dir"])]), 0)
        self.assertTrue(paths["output_path"].exists())
        self.assertFalse((paths["episode_dir"] / "tactile_flow.npz").exists())

    def test_png_source_change_during_computation_never_publishes_flow(self):
        paths = self._write_layout(self.episode.parent / "变化中", "by_episode")
        original_function = self.cv2.calcOpticalFlowFarneback
        touched = False
        def modify_source(*args, **kwargs):
            nonlocal touched
            if not touched:
                touched = True
                (paths["rgb_path"] / "finger1" / "frame_0000.png").unlink()
            return original_function(*args, **kwargs)
        with patch.object(self.cv2, "calcOpticalFlowFarneback", side_effect=modify_source):
            with self.assertRaisesRegex(RuntimeError, "source changed"):
                flow_module.save_episode_flow(**paths)
        self.assertFalse(paths["output_path"].exists())

    def test_explicit_output_cannot_overwrite_sources(self):
        self._write_episode()
        for source in (self.episode / "signals.csv", self.episode / "tactile_rgb.npz"):
            original = source.read_bytes()
            with self.assertRaisesRegex(ValueError, "must not overwrite"):
                flow_module.save_episode_flow(self.episode, output_path=source, overwrite=True)
            self.assertEqual(source.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
