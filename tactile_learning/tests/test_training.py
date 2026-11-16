"""Tests for metric accounting and training-time ablations."""
import unittest

import torch

from tactile.config import load_config
from tactile.engine import perturb_batch
from tactile.metrics import classification_metrics


class TrainingTests(unittest.TestCase):
    def test_confusion_and_macro_metrics(self):
        m = classification_metrics([0, 0, 1, 1], [0, 1, 1, 1], ["a", "b"])
        self.assertEqual(m["confusion_matrix"], [[1, 1], [0, 2]])
        self.assertEqual(m["accuracy"], .75)
        self.assertAlmostEqual(m["macro_f1"], (2/3 + .8)/2)

    def test_temporal_shuffle_keeps_flow_dt_correspondence(self):
        time = torch.arange(7).float()[None]
        batch = {"flow": time[:, None, :, None, None, None].expand(1, 3, 7, 2, 2, 2),
                 "flow_dt": time, "flow_mask": torch.ones(1, 3, 7, dtype=torch.bool),
                 "node_mask": torch.ones(1, 3, dtype=torch.bool),
                 "rgb_mask": torch.ones(1, 3, 2, dtype=torch.bool)}
        result = perturb_batch(batch, shuffle_time=True)
        self.assertTrue(torch.equal(result["flow"][0, 0, :, 0, 0, 0], result["flow_dt"][0]))
        self.assertTrue(torch.equal(result["flow"][0, 1], result["flow"][0, 2]))

    def test_dropout_preserves_valid_finger(self):
        batch = {"node_mask": torch.tensor([[1, 0, 0], [0, 1, 1]], dtype=torch.bool),
                 "rgb_mask": torch.ones(2, 3, 2, dtype=torch.bool),
                 "flow_mask": torch.ones(2, 3, 3, dtype=torch.bool)}
        result = perturb_batch(batch, finger_dropout=.9999)
        self.assertTrue(result["node_mask"].any(1).all())
        self.assertFalse((result["node_mask"] & ~batch["node_mask"]).any())

    def test_baseline_needs_no_feature_caches(self):
        config = load_config(variant="static_concat")
        self.assertFalse(config["data"]["use_flow"])
        self.assertFalse(config["data"]["use_geometry"])
        full = load_config()
        self.assertTrue(full["data"]["use_flow"])
        self.assertTrue(full["data"]["use_geometry"])


if __name__ == "__main__":
    unittest.main()
