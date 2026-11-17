"""Behavioral tests for masking, information flow and controlled model variants."""
from __future__ import annotations

import unittest

import torch
from torch import nn

from tactile.models import create_model


def example_batch(batch_size: int = 2) -> dict[str, torch.Tensor]:
    positions = torch.tensor([[0.04, 0.0, 0.02], [-0.02, 0.035, 0.02],
                              [-0.02, -0.035, 0.02]]).expand(batch_size, -1, -1)
    rotations = torch.eye(3).flatten().expand(batch_size, 3, -1)
    delta = positions[:, None, :, :] - positions[:, :, None, :]
    distance = delta.norm(dim=-1, keepdim=True)
    edge = torch.cat((delta, distance, torch.ones_like(distance)), dim=-1)
    return {
        "rgb": torch.randn(batch_size, 3, 2, 3, 32, 32),
        "rgb_mask": torch.ones(batch_size, 3, 2, dtype=torch.bool),
        "flow": torch.randn(batch_size, 3, 3, 2, 32, 32) * 0.01,
        "flow_mask": torch.ones(batch_size, 3, 3, dtype=torch.bool),
        "flow_dt": torch.full((batch_size, 3), 0.02),
        "sensor_pose": torch.cat((positions, rotations), dim=-1).clone(),
        "edge_attr": edge.clone(),
        "node_mask": torch.ones(batch_size, 3, dtype=torch.bool),
    }


def tiny_config(**updates):
    config = {"backbone": "tiny", "feature_dim": 16, "dropout": 0.0}
    config.update(updates)
    return config


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(31)

    def test_full_model_backward_reaches_every_branch(self):
        model = create_model(5, tiny_config())
        logits = model(example_batch())
        self.assertEqual(logits.shape, (2, 5))
        nn.functional.cross_entropy(logits, torch.tensor([1, 4])).backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, msg=name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), msg=name)
        self.assertGreater(model.graph.layers[0].edge.weight.grad.abs().sum().item(), 0)

    def test_invalid_nodes_frames_and_geometry_cannot_change_prediction(self):
        model = create_model(5, tiny_config()).eval()
        batch = example_batch()
        batch["node_mask"][:, 2] = False
        batch["rgb_mask"][:, :, 1] = False
        batch["flow_mask"][:, :, 2] = False
        changed = {key: value.clone() for key, value in batch.items()}
        changed["rgb"][:, :, 1] = float("nan")
        changed["rgb"][:, 2] = float("nan")
        changed["flow"][:, :, 2] = float("nan")
        changed["flow"][:, 2] = float("nan")
        changed["flow_dt"][:, 2] = float("nan")
        changed["sensor_pose"][:, 2] = float("nan")
        changed["edge_attr"][:, 2, :] = float("nan")
        changed["edge_attr"][:, :, 2] = float("nan")
        with torch.no_grad():
            reference, result = model(batch), model(changed)
        torch.testing.assert_close(reference, result, rtol=0, atol=0)

    def test_appending_padding_does_not_update_gru(self):
        model = create_model(5, tiny_config()).eval()
        batch = example_batch()
        changed = {key: value.clone() for key, value in batch.items()}
        changed["flow"] = torch.cat((batch["flow"], torch.full((2, 3, 2, 2, 32, 32),
                                                             float("nan"))), dim=2)
        changed["flow_mask"] = torch.cat((batch["flow_mask"],
                                          torch.zeros(2, 3, 2, dtype=torch.bool)), dim=2)
        changed["flow_dt"] = torch.cat((batch["flow_dt"], torch.full((2, 2), float("nan"))), dim=1)
        with torch.no_grad():
            torch.testing.assert_close(model(batch), model(changed), rtol=0, atol=0)

    def test_graph_is_invariant_to_consistent_node_permutation(self):
        model = create_model(5, tiny_config()).eval()
        batch = example_batch()
        batch["node_mask"][0, 2] = False
        permutation = torch.tensor([2, 0, 1])
        changed = {key: value.clone() for key, value in batch.items()}
        for key in ("rgb", "rgb_mask", "flow", "flow_mask", "sensor_pose", "node_mask"):
            changed[key] = batch[key][:, permutation]
        changed["edge_attr"] = batch["edge_attr"][:, permutation][:, :, permutation]
        with torch.no_grad():
            torch.testing.assert_close(model(batch), model(changed), atol=2e-6, rtol=2e-5)

    def test_single_finger_cannot_read_other_fingers(self):
        model = create_model(5, tiny_config(fusion="single", single_finger=1))
        batch = example_batch()
        changed = {key: value.clone() for key, value in batch.items()}
        for key in ("rgb", "flow", "sensor_pose"):
            changed[key][:, [0, 2]] = float("nan")
        changed["edge_attr"][:] = float("nan")
        # Training mode also verifies that irrelevant frames never enter the CNN.
        torch.testing.assert_close(model(batch), model(changed), rtol=0, atol=0)

    def test_baselines_and_ablations_run_without_disabled_inputs(self):
        for options in ({"fusion": "concat"}, {"fusion": "mean"},
                        {"fusion": "single"}, {"use_flow": False},
                        {"use_alignment": False}, {"use_edge_geometry": False},
                        {"use_flow": False, "use_alignment": False,
                         "use_edge_geometry": False}):
            with self.subTest(options=options):
                model = create_model(5, {"model": tiny_config(**options)}).eval()
                batch = example_batch()
                if options.get("use_flow") is False:
                    for key in ("flow", "flow_mask", "flow_dt"):
                        del batch[key]
                if options.get("use_alignment") is False:
                    del batch["sensor_pose"]
                if options.get("use_edge_geometry") is False:
                    del batch["edge_attr"]
                logits = model(batch)
                self.assertEqual(logits.shape, (2, 5))
                self.assertTrue(torch.isfinite(logits).all())

    def test_empty_flow_history_is_supported(self):
        model = create_model(5, tiny_config()).eval()
        batch = example_batch()
        batch["flow_mask"][:] = False
        batch["flow"][:] = float("nan")
        batch["flow_dt"][:] = float("nan")
        self.assertTrue(torch.isfinite(model(batch)).all())
        batch["flow"] = batch["flow"][:, :, :0]
        batch["flow_mask"] = batch["flow_mask"][:, :, :0]
        batch["flow_dt"] = batch["flow_dt"][:, :0]
        self.assertTrue(torch.isfinite(model(batch)).all())

    def test_all_invalid_episode_has_clear_error(self):
        batch = example_batch()
        batch["node_mask"][0] = False
        with self.assertRaisesRegex(ValueError, "valid contact"):
            create_model(5, tiny_config())(batch)

    def test_valid_edges_affect_graph_output(self):
        model = create_model(5, tiny_config()).eval()
        batch = example_batch()
        changed = {key: value.clone() for key, value in batch.items()}
        changed["edge_attr"][:, 0, 1, 0] += 0.1
        with torch.no_grad():
            difference = (model(batch) - model(changed)).abs().max().item()
        self.assertGreater(difference, 1e-7)


    def test_hierarchical_gnn_local_and_global_graph_receive_gradients(self):
        model = create_model(5, tiny_config(
            fusion="graph",
            use_flow=True,
            use_local_graph=True,
            use_alignment=False,
            use_edge_geometry=False,
        ))
        batch = example_batch()
        logits = model(batch)
        self.assertEqual(logits.shape, (2, 5))
        nn.functional.cross_entropy(logits, torch.tensor([1, 4])).backward()

        self.assertIsNotNone(model.local_graph.modality_embedding.grad)
        self.assertTrue(torch.isfinite(model.local_graph.modality_embedding.grad).all())
        self.assertGreater(
            model.local_graph.layers[0].attention.grad.abs().sum().item(), 0
        )
        self.assertGreater(
            model.graph.layers[0].attention.grad.abs().sum().item(), 0
        )

    def test_hierarchical_gnn_masks_missing_flow_modality_per_finger(self):
        model = create_model(5, tiny_config(
            fusion="graph",
            use_flow=True,
            use_local_graph=True,
            use_alignment=False,
            use_edge_geometry=False,
        )).eval()
        batch = example_batch()
        batch["flow_mask"][:, 2] = False
        batch["flow"][:, 2] = float("nan")
        with torch.no_grad():
            logits = model(batch)
        self.assertEqual(logits.shape, (2, 5))
        self.assertTrue(torch.isfinite(logits).all())

    def test_original_dynamic_concat_has_no_local_graph(self):
        model = create_model(5, tiny_config(fusion="concat", use_flow=True))
        self.assertIsNone(model.local_graph)



if __name__ == "__main__":
    unittest.main()
