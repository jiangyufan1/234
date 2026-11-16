"""Exercise rejection and rollback with real MuJoCo steps, not collision mocks."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import collect_mcb_gelsight_dataset as adapter
import collect_mcb_position_pose as core
from sensor_collision_guard import SensorCollisionError, SensorCollisionGuard


def scene():
    xml = '<mujoco><option timestep=".001" gravity="0 0 0"/><worldbody>'
    for finger, x in enumerate((0, .05, .2), 1):
        xml += (f'<body pos="{x} 0 0"><freejoint/>'
                f'<geom name="finger{finger}_gel" type="box" size=".01 .01 .01" '
                'mass=".01" contype="2" conaffinity="4"/></body>')
    model = mujoco.MjModel.from_xml_string(xml + '</worldbody></mujoco>')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


class AdapterSensorSafetyTests(unittest.TestCase):
    def setUp(self):
        self.native_step = mujoco.mj_step

    def tearDown(self):
        mujoco.mj_step = self.native_step

    def collector(self, episode, capture=None):
        collector = SimpleNamespace(mujoco=mujoco, run_episode=episode,
                                    episode_output_complete=lambda *args: True)
        if capture is not None:
            collector.capture_gelsight_images = capture
        adapter.install_sensor_safety(collector, .001)
        return collector

    def test_safe_episode_keeps_samples_and_certifies_checked_states(self):
        model, data = scene()
        samples = [{"contacts": [True, False, False]}]
        def run(**kwargs):
            mujoco.mj_step(model, data, nstep=3)
            return {"valid": True, "reason": "ok", "metadata": {}, "samples": samples}
        collector = self.collector(run)
        result = collector.run_episode(model=model, data=data)
        self.assertIs(result["samples"], samples)
        safety = result["metadata"]["sensor_safety"]
        self.assertTrue(safety["passed"])
        self.assertEqual(safety["physics_steps"], 3)
        self.assertGreaterEqual(safety["minimum_distance_m"], .001)

    def test_bulk_steps_stop_before_contact_and_discard_prior_samples(self):
        model, data = scene()
        data.qvel[6] = -1
        def run(**kwargs):
            # With one unchecked 100-step call the second box crosses through
            # the first and separates again. Inspecting every step must stop it.
            mujoco.mj_step(model, data, 100)
            self.fail("Unsafe bulk motion must not finish")
        collector = self.collector(run)
        result = collector.run_episode(model=model, data=data, episode_idx=7, pose_cfg={"id": 3})
        self.assertFalse(result["valid"])
        self.assertEqual(result["samples"], [])
        self.assertEqual(result["metadata"]["episode_index"], 7)
        safety = result["metadata"]["sensor_safety"]
        self.assertTrue(safety["discarded_entire_episode"])
        self.assertTrue(safety["restored_safe_state"])
        self.assertTrue(SensorCollisionGuard(model).inspect(data)["safe"])
        self.assertLess(data.time, .1)
        self.assertEqual(data.ncon, 0)  # collision masks cannot provide this protection

    def test_inner_catch_cannot_turn_rejected_frame_into_saved_samples(self):
        model, data = scene()
        calls = []
        def run(**kwargs):
            data.qpos[7] = .019
            try:
                collector.capture_gelsight_images(model, data)
            except SensorCollisionError:
                pass
            return {"valid": True, "reason": "ok", "metadata": {}, "samples": ["contaminated"]}
        collector = self.collector(run, capture=lambda *args: calls.append(True))
        result = collector.run_episode(model=model, data=data)
        self.assertFalse(result["valid"])
        self.assertEqual(result["samples"], [])
        self.assertEqual(calls, [])
        self.assertFalse(result["metadata"]["optical_flow_saved"])

    def test_no_episode_scope_does_not_interrupt_calibration(self):
        model, data = scene()
        data.qvel[6] = -1
        self.collector(lambda **kwargs: None)
        mujoco.mj_step(model, data, 60)
        self.assertAlmostEqual(data.time, .06)

    def test_unrelated_renderer_errors_propagate(self):
        def run(**kwargs):
            raise RuntimeError("renderer failed")
        collector = self.collector(run)
        with self.assertRaisesRegex(RuntimeError, "renderer failed"):
            collector.run_episode()

    def test_resume_requires_safety_certificate_even_for_zero_frame_rejections(self):
        collector = self.collector(lambda **kwargs: None)
        self.assertFalse(collector.episode_output_complete(None, {"valid": False}, {}))
        policy = collector.SENSOR_SAFETY_POLICY
        self.assertTrue(collector.episode_output_complete(None,
                        {"valid": False, "sensor_safety": {"policy": policy}}, {}))
        self.assertFalse(collector.episode_output_complete(None,
                         {"valid": True, "sensor_safety": {"policy": policy}}, {}))

    def test_clearance_cli_rejects_disabled_or_nonfinite_protection(self):
        self.assertEqual(core.parse_args([]).sensor_clearance_mm, 1)
        for value in ("0", "-1", "nan", "inf"):
            with self.subTest(value=value), self.assertRaises(SystemExit):
                core.parse_args(["--sensor-clearance-mm", value])


if __name__ == "__main__":
    unittest.main()
