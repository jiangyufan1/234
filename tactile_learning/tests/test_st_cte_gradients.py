
import torch
from tactile.models.classifier import create_model


def make_batch(batch_size=2):
    b, n, tr, tf, h, w = batch_size, 3, 4, 3, 64, 48
    times = torch.tensor([[0.00, 0.04, 0.08, 0.12]]).repeat(b, 1)
    starts, ends = times[:, :-1].clone(), times[:, 1:].clone()
    return {
        "rgb": torch.randn(b, n, tr, 3, h, w),
        "flow": torch.randn(b, n, tf, 2, h, w) * 0.05,
        "rgb_mask": torch.ones(b, n, tr, dtype=torch.bool),
        "flow_mask": torch.ones(b, n, tf, dtype=torch.bool),
        "flow_dt": ends - starts,
        "rgb_times": times,
        "flow_start": starts,
        "flow_end": ends,
        "node_mask": torch.ones(b, n, dtype=torch.bool),
        "label": torch.arange(b, dtype=torch.long) % 2,
        "episode_id": [f"synthetic-{i}" for i in range(b)],
    }


def has_nonzero_grad(module):
    vals = [p.grad.detach().abs().sum().item()
            for p in module.parameters()
            if p.requires_grad and p.grad is not None]
    return bool(vals) and any(v > 0 for v in vals)


def test_full_model_forward_backward_gradients_and_optimizer_update():
    torch.manual_seed(2026)
    config = {
        "architecture": "st_cte_hgnn",
        "feature_dim": 32,
        "spatial_dim": 32,
        "spatial_graph_layers": 1,
        "backbone": "tiny",
        "pretrained": False,
        "dropout": 0.0,
        "cte_num_atoms": 4,
        "cte_sigma_seconds": 0.05,
        "use_flow": True,
        "use_alignment": False,
        "fusion": "graph",
        "use_edge_geometry": False,
        "use_local_graph": False,
    }
    model = create_model(2, config)
    batch = make_batch()
    logits, aux = model.forward_with_aux(batch)
    assert logits.shape == (2, 2)
    loss = torch.nn.functional.cross_entropy(logits, batch["label"])
    aux_names = {"trend_smoothness", "event_sparsity", "decomposition_reconstruction"}
    terms = [v for k, v in aux.items()
             if k in aux_names or (k.startswith("flow_") and k[5:] in aux_names)]
    if terms:
        loss = loss + torch.stack(terms).sum()
    assert torch.isfinite(loss)
    loss.backward()

    for name, module in (
        ("spatial CNN/GAT", model.spatial),
        ("RGB/Flow encoder", model.encoder),
        ("CTE temporal HGNN", model.temporal),
        ("spatiotemporal fusion", model.fusion),
        ("classifier", model.classifier),
    ):
        assert has_nonzero_grad(module), f"{name} has no nonzero gradient"

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    parameter = next(model.spatial.parameters())
    before = parameter.detach().clone()
    optimizer.step()
    assert not torch.equal(before, parameter.detach())


def test_online_prefix_interface():
    torch.manual_seed(2027)
    config = {
        "architecture": "st_cte_hgnn",
        "feature_dim": 32,
        "spatial_dim": 32,
        "spatial_graph_layers": 1,
        "backbone": "tiny",
        "pretrained": False,
        "dropout": 0.0,
        "cte_num_atoms": 4,
        "cte_sigma_seconds": 0.05,
    }
    model = create_model(2, config).eval()
    batch = make_batch(batch_size=1)
    with torch.inference_mode():
        result = model.forward_online(batch, query_time=0.04)
    assert result["logits"].shape == (1, 2)
    assert result["probabilities"].shape == (1, 2)
    assert result["prediction"].shape == (1,)
    assert torch.allclose(result["probabilities"].sum(-1), torch.ones(1), atol=1e-5)
