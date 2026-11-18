import torch

from tactile.models.cte_hgnn import CTEHGNN


def make_model():
    torch.manual_seed(123)
    return CTEHGNN(
        dim=8,
        num_classes=5,
        num_atoms=4,
        dropout=0.0,
    ).eval()


def test_future_observations_do_not_change_causal_prediction():
    torch.manual_seed(42)

    model = make_model()

    B, N, Tr, Tf, D = 1, 3, 4, 5, 8

    rgb = torch.randn(B, N, Tr, D)
    flow = torch.randn(B, N, Tf, D)

    rgb_times = torch.tensor([[0.01, 0.03, 0.05, 0.07]])
    flow_start = torch.tensor([[0.00, 0.02, 0.04, 0.06, 0.08]])
    flow_end = flow_start + 0.02

    rgb_mask = torch.ones(B, N, Tr, dtype=torch.bool)
    flow_mask = torch.ones(B, N, Tf, dtype=torch.bool)

    query = torch.tensor([0.05])

    # Full episode, but model must only see the causal prefix.
    with torch.no_grad():
        full_logits = model(
            rgb, flow,
            rgb_times, flow_start, flow_end,
            rgb_mask, flow_mask,
            query_time=query,
            return_aux=False,
        )

        # Physically truncate all observations after query_time.
        prefix_logits = model(
            rgb[:, :, :3],
            flow[:, :, :2],
            rgb_times[:, :3],
            flow_start[:, :2],
            flow_end[:, :2],
            rgb_mask[:, :, :3],
            flow_mask[:, :, :2],
            query_time=query,
            return_aux=False,
        )

    assert torch.allclose(
        full_logits, prefix_logits, atol=1e-5, rtol=1e-5
    )


def test_rgb_only_prediction_before_first_flow_completion():
    torch.manual_seed(43)

    model = make_model()

    B, N, Tr, Tf, D = 1, 3, 3, 2, 8

    rgb = torch.randn(B, N, Tr, D)
    flow = torch.randn(B, N, Tf, D)

    rgb_times = torch.tensor([[0.01, 0.03, 0.05]])
    flow_start = torch.tensor([[0.00, 0.02]])
    flow_end = torch.tensor([[0.02, 0.04]])

    rgb_mask = torch.ones(B, N, Tr, dtype=torch.bool)
    flow_mask = torch.ones(B, N, Tf, dtype=torch.bool)

    # At t=0.01, no Flow interval has completed.
    with torch.no_grad():
        logits = model(
            rgb, flow,
            rgb_times, flow_start, flow_end,
            rgb_mask, flow_mask,
            query_time=torch.tensor([0.01]),
            return_aux=False,
        )

    assert logits.shape == (B, 5)
    assert torch.isfinite(logits).all()


def test_future_feature_values_cannot_change_prefix_prediction():
    torch.manual_seed(44)

    model = make_model()

    B, N, D = 1, 3, 8
    Tr, Tf = 4, 5

    rgb = torch.randn(B, N, Tr, D)
    flow = torch.randn(B, N, Tf, D)

    rgb_times = torch.tensor([[0.01, 0.03, 0.05, 0.07]])
    flow_start = torch.tensor([[0.00, 0.02, 0.04, 0.06, 0.08]])
    flow_end = flow_start + 0.02

    rm = torch.ones(B, N, Tr, dtype=torch.bool)
    fm = torch.ones(B, N, Tf, dtype=torch.bool)

    rgb_changed = rgb.clone()
    flow_changed = flow.clone()

    rgb_changed[:, :, 3:] = torch.randn_like(rgb_changed[:, :, 3:]) * 100
    flow_changed[:, :, 2:] = torch.randn_like(flow_changed[:, :, 2:]) * 100

    query = torch.tensor([0.05])

    with torch.no_grad():
        a = model(
            rgb, flow, rgb_times, flow_start, flow_end,
            rm, fm, query_time=query, return_aux=False,
        )

        b = model(
            rgb_changed, flow_changed,
            rgb_times, flow_start, flow_end,
            rm, fm, query_time=query, return_aux=False,
        )

    assert torch.allclose(a, b, atol=1e-5, rtol=1e-5)
