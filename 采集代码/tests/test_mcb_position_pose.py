import json
from pathlib import Path
import sys
import tempfile
import unittest

# Tests can be launched from the workspace root after code consolidation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import numpy as np

import collect_mcb_gelsight_dataset as adapter
import collect_mcb_position_pose as sweep


class SamplingTests(unittest.TestCase):
    def test_independent_cartesian_product_and_units(self):
        args = sweep.parse_args([])
        self.assertEqual(args.layout, "by_type")
        configs, dims = sweep.make_configs(args)
        self.assertEqual(len(configs), 7 * 7 * 7)
        self.assertEqual(dims["episodes"], 343)
        for key in ("position_group", "pose_group", "contact_group"):
            self.assertEqual(len({cfg[key] for cfg in configs}), 7)
        self.assertFalse(configs[0]["follow_object_orientation"])
        self.assertIn([0.01, 0.0, 0.0], [c["object_position_offset_world_m"] for c in configs])
        self.assertIn([0.005, 0.0, 0.0], [c["grasp_offset_object_m"] for c in configs])

    def test_small_pilot_varies_all_three_factors(self):
        configs, dims = sweep.make_configs(sweep.parse_args(["--max-configs", "2", "--repeats", "3"]))
        self.assertEqual(dims["episodes"], 6)
        for key in ("object_position_offset_world_m", "object_quat_wxyz", "grasp_offset_object_m"):
            self.assertNotEqual(configs[0][key], configs[1][key])

    def test_rotation_convention(self):
        quat = np.array(sweep.rpy_quaternion([0, 0, 90]))
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, quat)
        np.testing.assert_allclose(rotation.reshape(3, 3) @ [1, 0, 0], [0, 1, 0], atol=1e-12)

    def test_bad_numbers_and_counts(self):
        with self.assertRaises(ValueError):
            sweep.make_configs(sweep.parse_args(["--position-mm", "nan", "0", "0"]))
        with self.assertRaises(SystemExit):
            sweep.parse_args(["--max-configs", "0"])

    def test_resume_requires_identical_plan_and_never_overwrites_unknown_output(self):
        configs, dims = sweep.make_configs(sweep.parse_args(["--max-configs", "2"]))
        plan = {"configs": configs, "dimensions": dims}
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "run"
            sweep.save_plan(folder, plan, resume=False)
            original = (folder / "collection_plan.json").read_bytes()
            with self.assertRaises(ValueError):
                sweep.save_plan(folder, plan, resume=False)
            sweep.save_plan(folder, plan, resume=True)
            changed = json.loads(json.dumps(plan))
            changed["configs"][0]["gripper_roll_deg"] = 60
            with self.assertRaises(ValueError):
                sweep.save_plan(folder, changed, resume=True)
            self.assertEqual((folder / "collection_plan.json").read_bytes(), original)

    def test_grouped_plan_is_saved_with_its_file_format(self):
        configs, dims = sweep.make_configs(sweep.parse_args(["--max-configs", "2"]))
        plan = {"configs": configs, "dimensions": dims, "output_layout": "by_type"}
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "run"
            sweep.save_plan(folder, plan, resume=False)
            self.assertTrue((folder / "metadata_json" / "collection_plan.json").is_file())
            self.assertTrue((folder / "signals_csv" / "collection_plan.csv").is_file())
            self.assertFalse((folder / "collection_plan.json").exists())
            sweep.save_plan(folder, plan, resume=True)


class BasePositionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = adapter.load_collector(adapter.DEFAULT_COLLECTOR, skip_taxim=True)
        cls.base.OBJECT_BODY = "mcb_part"
        cls.base.OBJECT_FREEJOINT = "mcb_part_free"
        cls.base.XY_RANDOM_RANGE_M = 0.0

    def test_probe_does_not_poison_real_taxim_import(self):
        prior = sys.modules.get("TaximSensor")
        adapter.load_collector(adapter.DEFAULT_COLLECTOR, skip_taxim=True)
        self.assertIs(sys.modules.get("TaximSensor"), prior)
        real = adapter.load_collector(adapter.DEFAULT_COLLECTOR, skip_taxim=False)
        self.assertIsNot(real.TaximSensor, object)

    def test_world_translation_is_applied_to_freejoint(self):
        model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
          <body name="mcb_part" pos="0.4 -0.1 0.84"><freejoint name="mcb_part_free"/>
          <geom type="sphere" size="0.02" mass="1"/></body></worldbody></mujoco>''')
        data = mujoco.MjData(model)
        cfg = {"pose_group": "check", "object_quat_wxyz": [1, 0, 0, 0],
               "object_position_offset_world_m": [0.01, -0.02, 0.03]}
        command = self.base.randomize_object_pose(model, data, np.random.default_rng(4), cfg)
        np.testing.assert_allclose(data.qpos[:3], [0.41, -0.12, 0.87])
        np.testing.assert_allclose(command["initial_position_m"], data.qpos[:3])
        # Contact offset is a separate control; it cannot move the object.
        second = mujoco.MjData(model)
        cfg["grasp_offset_object_m"] = [0.005, 0, 0]
        self.base.randomize_object_pose(model, second, np.random.default_rng(4), cfg)
        np.testing.assert_allclose(second.qpos[:3], data.qpos[:3])

    def test_fixture_reset_does_not_restore_previous_episode(self):
        base = adapter.load_collector(adapter.DEFAULT_COLLECTOR, skip_taxim=True)
        xml = '''<mujoco><option gravity="0 0 0"/><asset>
          <mesh name="part" vertex="0 0 0  .02 0 0  0 .02 0  0 0 .02"/>
          </asset><worldbody><body name="mcb_part" pos="0.4 0 0.84">
          <freejoint name="mcb_part_free"/>
          <geom name="mcb_part_visual" type="mesh" mesh="part" contype="0" conaffinity="0"/>
          <geom name="mcb_part_collision" type="mesh" mesh="part"/>
          </body></worldbody></mujoco>'''
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "part.xml"
            path.write_text(xml, encoding="utf-8")
            # A known failed episode must keep its useful diagnostic reason.
            base.run_episode = lambda *a, **k: {"valid": False, "reason": "safe_ik_failed",
                                                "metadata": {}, "samples": []}
            adapter.configure_collector(base, path, "part", Path(directory), [{}],
                                        1, 4, False, False, True)
            base.XY_RANDOM_RANGE_M = 0.0
            try:
                model = mujoco.MjModel.from_xml_string(xml)
                data = mujoco.MjData(model)
                cfg = {"pose_group": "a", "object_quat_wxyz": sweep.rpy_quaternion([0, 0, 45]),
                       "object_position_offset_world_m": [0.01, 0.02, 0.03]}
                base.reset_home(model, data)
                base.randomize_object_pose(model, data, np.random.default_rng(4), cfg)
                base.mujoco.mj_step(model, data)
                self.assertFalse(np.allclose(data.qpos, model.qpos0))
                base.reset_home(model, data)
                base.mujoco.mj_step(model, data)
                np.testing.assert_allclose(data.qpos, model.qpos0, atol=1e-12)
                failed = base.run_episode()
                self.assertEqual(failed["reason"], "safe_ik_failed")
                self.assertEqual(failed["metadata"]["reason"], "safe_ik_failed")
                self.assertFalse(failed["metadata"]["valid"])
            finally:
                base.mujoco.mj_step = base._mcb_base_mj_step


if __name__ == "__main__":
    unittest.main()
