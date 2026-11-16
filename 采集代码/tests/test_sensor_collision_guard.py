import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import numpy as np

from sensor_collision_guard import SensorCollisionError, SensorCollisionGuard


def model_and_data(second_x=0.04, *, masks="0 0", housings=False, pair=False, housing_offset=0):
    contype, affinity = masks.split()
    geometries = []
    for index, x in enumerate((0, second_x, 0.2), 1):
        housing = (
            f'<geom name="finger{index}_gelsight_housing" type="box" '
            f'pos="{housing_offset if index == 1 else 0} 0 0" '
            'size=".01 .01 .01" contype="0" conaffinity="0"/>'
        ) if housings else ""
        geometries.append(
            f'<body pos="{x} 0 0"><freejoint/>'
            f'<geom name="finger{index}_gel" type="box" size=".01 .01 .01" '
            f'contype="{contype}" conaffinity="{affinity}" mass=".01"/>{housing}</body>'
        )
    contact = (
        '<contact><pair geom1="finger1_gel" geom2="finger2_gel" margin=".004"/></contact>'
    ) if pair else ""
    xml = '<mujoco><option gravity="0 0 0"/><worldbody>' + "".join(geometries) + '</worldbody>' + contact + '</mujoco>'
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


class SensorCollisionGuardTests(unittest.TestCase):
    def test_disabled_collisions_still_measure_real_distance(self):
        for masks in ("0 0", "2 4"):
            with self.subTest(masks=masks):
                model, data = model_and_data(masks=masks)
                report = SensorCollisionGuard(model).inspect(data)
                self.assertTrue(report["safe"])
                self.assertEqual(data.ncon, 0)
                self.assertAlmostEqual(report["minimum_distance_m"], 0.02)
                self.assertEqual(report["closest_pair"], ["finger1_gel", "finger2_gel"])
                self.assertEqual(report["checked_pairs"], 3)
                json.dumps(report, allow_nan=False)

    def test_disabled_collisions_still_detect_penetration(self):
        model, data = model_and_data(0.019, masks="2 4")
        guard = SensorCollisionGuard(model)
        report = guard.inspect(data)
        self.assertEqual(data.ncon, 0)
        self.assertFalse(report["safe"])
        self.assertAlmostEqual(report["minimum_distance_m"], -0.001)
        self.assertEqual(report["reason"], "sensor_surface_collision")
        with self.assertRaises(SensorCollisionError) as raised:
            guard.require_safe(data)
        self.assertEqual(raised.exception.report, report)

    def test_same_finger_overlapping_housing_is_ignored(self):
        model, data = model_and_data(housings=True)
        report = SensorCollisionGuard(model).inspect(data)
        self.assertTrue(report["safe"])
        self.assertAlmostEqual(report["minimum_distance_m"], 0.02)
        self.assertEqual(report["checked_pairs"], 12)

    def test_housing_to_other_finger_gel_is_checked(self):
        model, data = model_and_data(0.03, housings=True, housing_offset=0.0095)
        report = SensorCollisionGuard(model).inspect(data)
        self.assertFalse(report["safe"])
        self.assertAlmostEqual(report["minimum_distance_m"], 0.0005)
        self.assertIn("finger1_gelsight_housing", report["closest_pair"])
        self.assertEqual(report["reason"], "sensor_clearance_below_threshold")

    def test_actual_positive_margin_contact_rejects_safe_clearance(self):
        model, data = model_and_data(0.023, pair=True)
        self.assertGreater(data.ncon, 0)
        report = SensorCollisionGuard(model, clearance_m=0.001).inspect(data)
        self.assertFalse(report["safe"])
        self.assertGreater(report["minimum_distance_m"], report["threshold_m"])
        self.assertEqual(report["reason"], "cross_sensor_contact")
        self.assertGreater(report["cross_sensor_contacts"][0]["distance_m"], 0)

    def test_positive_threshold_accepts_equality_and_rejects_below(self):
        model, data = model_and_data(0.025)
        exact = SensorCollisionGuard(model, clearance_m=0).inspect(data)["minimum_distance_m"]
        self.assertTrue(SensorCollisionGuard(model, clearance_m=exact).inspect(data)["safe"])
        self.assertFalse(SensorCollisionGuard(model, clearance_m=exact + 1e-7).inspect(data)["safe"])

    def test_zero_threshold_still_rejects_touching(self):
        model, data = model_and_data(0.02)
        report = SensorCollisionGuard(model, clearance_m=0).inspect(data)
        self.assertFalse(report["safe"])
        self.assertEqual(report["minimum_distance_m"], 0)

    def test_invalid_thresholds_fail_early(self):
        model, _ = model_and_data()
        for value in (-0.001, float("nan"), float("inf"), -float("inf"), None, "abc"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "finite and nonnegative"):
                SensorCollisionGuard(model, clearance_m=value)

    def test_missing_gels_refuses_to_disable_guard(self):
        model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom name="finger1_gel" type="box" size=".01 .01 .01"/></worldbody></mujoco>')
        with self.assertRaisesRegex(ValueError, "finger2_gel, finger3_gel"):
            SensorCollisionGuard(model)

    def test_non_box_sensor_refuses_unsupported_geometry(self):
        xml = '<mujoco><worldbody>' + ''.join(
            f'<geom name="finger{i}_gel" type="sphere" size=".01"/>' for i in (1, 2, 3)
        ) + '</worldbody></mujoco>'
        model = mujoco.MjModel.from_xml_string(xml)
        with self.assertRaisesRegex(ValueError, "must be a box"):
            SensorCollisionGuard(model)

    def test_nonfinite_sensor_transform_rejects_without_distance_query(self):
        model, data = model_and_data()
        data.geom_xpos[0, 0] = np.nan
        with patch("sensor_collision_guard.mujoco.mj_geomDistance") as distance:
            report = SensorCollisionGuard(model).inspect(data)
        distance.assert_not_called()
        self.assertFalse(report["safe"])
        self.assertEqual(report["reason"], "nonfinite_sensor_transform")
        json.dumps(report, allow_nan=False)

    def test_spurious_native_zero_does_not_reject_provably_separated_boxes(self):
        model, data = model_and_data()
        with patch("sensor_collision_guard.mujoco.mj_geomDistance", return_value=0):
            report = SensorCollisionGuard(model).inspect(data)
        self.assertTrue(report["safe"])
        self.assertTrue(report["distance_is_lower_bound"])
        self.assertAlmostEqual(report["minimum_distance_m"], .02)

    def test_invalid_box_sizes_refuse_certification(self):
        model, data = model_and_data()
        model.geom_size[model.geom("finger1_gel").id, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "sizes must be finite"):
            SensorCollisionGuard(model)


if __name__ == "__main__":
    unittest.main()
