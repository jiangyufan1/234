from copy import deepcopy

import torch

from tactile.models.temporal_graph import PointIntervalGraphBuilder
from tactile.models.cte_hgnn import TemporalHeteroMessagePassing
from tactile.models.dynamic_routing import DynamicPseudoRouter


def make_graph(B=2, N=3, Tr=4, Tf=7, D=8):
    names = (
        "rgb_trend", "rgb_event",
        "flow_trend", "flow_event",
    )

    x = {
        "rgb_trend": torch.randn(B, N, Tr, D, requires_grad=True),
        "rgb_event": torch.randn(B, N, Tr, D, requires_grad=True),
        "flow_trend": torch.randn(B, N, Tf, D, requires_grad=True),
        "flow_event": torch.randn(B, N, Tf, D, requires_grad=True),
    }

    m = {
        "rgb_trend": torch.ones(B, N, Tr, dtype=torch.bool),
        "rgb_event": torch.ones(B, N, Tr, dtype=torch.bool),
        "flow_trend": torch.ones(B, N, Tf, dtype=torch.bool),
        "flow_event": torch.ones(B, N, Tf, dtype=torch.bool),
    }

    # Generate timestamps from the requested batch and sequence lengths.
    rt = (
        torch.arange(Tr, dtype=torch.float32) * 0.02 + 0.01
    ).unsqueeze(0).expand(B, Tr).clone()

    fs = (
        torch.arange(Tf, dtype=torch.float32) * 0.01
    ).unsqueeze(0).expand(B, Tf).clone()

    fe = fs + 0.01

    graph = PointIntervalGraphBuilder()(
        x, m, rt, fs, fe
    )

    return graph, x


def test_all_four_node_streams_receive_gradients():
    graph, x = make_graph()
    model = TemporalHeteroMessagePassing(8, dropout=0.0)

    output = model(graph)
    loss = sum(v.square().mean() for v in output.values())
    loss.backward()

    for name in x:
        assert x[name].grad is not None, name
        assert torch.isfinite(x[name].grad).all(), name
        assert x[name].grad.abs().sum() > 0, name


def test_bidirectional_messages_change_both_modalities():
    graph, _ = make_graph(B=1, N=1, Tr=3, Tf=5, D=8)
    model = TemporalHeteroMessagePassing(8, dropout=0.0).eval()

    with torch.no_grad():
        output = model(graph)

    for name in (
        "rgb_trend", "rgb_event",
        "flow_trend", "flow_event",
    ):
        assert not torch.allclose(
            output[name], graph.node_x[name]
        )


def test_edge_compatibility_changes_message_passing():
    graph, _ = make_graph(B=1, N=1, Tr=3, Tf=4, D=8)
    model = TemporalHeteroMessagePassing(8, dropout=0.0).eval()

    altered = deepcopy(graph)
    altered.edge_attr[..., 4] = (
        altered.edge_attr[..., 4] * 0.2
    )

    with torch.no_grad():
        original_output = model(graph)
        altered_output = model(altered)

    differences = [
        (original_output[k] - altered_output[k]).abs().max()
        for k in original_output
    ]

    assert max(float(v) for v in differences) > 1e-7


def test_router_preserves_finger_independence():
    torch.manual_seed(7)

    router = DynamicPseudoRouter(8, heads=4, dropout=0.0).eval()

    rgb = torch.randn(1, 3, 8)
    flow = torch.randn(1, 3, 8)
    mask = torch.ones(1, 3, dtype=torch.bool)
    edge = torch.randn(1, 3, 5)

    changed = rgb.clone()
    changed[:, 0] += 10.0

    with torch.no_grad():
        out1 = router(rgb, mask, flow, mask, edge)
        out2 = router(changed, mask, flow, mask, edge)

    assert torch.allclose(out1[0][:, 1:], out2[0][:, 1:])
    assert torch.allclose(out1[1][:, 1:], out2[1][:, 1:])


def test_router_handles_missing_modality_without_nan():
    router = DynamicPseudoRouter(8, heads=4, dropout=0.0)

    rgb = torch.randn(2, 3, 8)
    flow = torch.randn(2, 3, 8)

    rgb_mask = torch.tensor([
        [True, True, False],
        [False, False, False],
    ])

    flow_mask = torch.tensor([
        [False, True, True],
        [False, False, False],
    ])

    edge = torch.zeros(2, 3, 5)

    rr, ff, aux = router(
        rgb, rgb_mask, flow, flow_mask, edge
    )

    assert torch.isfinite(rr).all()
    assert torch.isfinite(ff).all()
    assert torch.isfinite(aux["rgb_gate"]).all()
    assert torch.isfinite(aux["flow_gate"]).all()

    assert torch.all(rr[~rgb_mask] == 0)
    assert torch.all(ff[~flow_mask] == 0)
