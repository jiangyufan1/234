import torch

from tactile.models.temporal_graph import PointIntervalGraphBuilder


def make_inputs(Tr, Tf, B=1, N=3, D=8):
    node_x = {
        "rgb_trend": torch.randn(B, N, Tr, D),
        "rgb_event": torch.randn(B, N, Tr, D),
        "flow_trend": torch.randn(B, N, Tf, D),
        "flow_event": torch.randn(B, N, Tf, D),
    }

    node_mask = {
        "rgb_trend": torch.ones(B, N, Tr, dtype=torch.bool),
        "rgb_event": torch.ones(B, N, Tr, dtype=torch.bool),
        "flow_trend": torch.ones(B, N, Tf, dtype=torch.bool),
        "flow_event": torch.ones(B, N, Tf, dtype=torch.bool),
    }

    return node_x, node_mask


def test_point_interval_causal_rule():
    B, N, Tr, Tf, D = 1, 1, 3, 3, 4

    x, m = make_inputs(Tr, Tf, B, N, D)

    rt = torch.tensor([[0.00, 0.02, 0.04]])
    fs = torch.tensor([[0.00, 0.02, 0.04]])
    fe = torch.tensor([[0.02, 0.04, 0.06]])
    query = torch.tensor([0.04])

    g = PointIntervalGraphBuilder()(
        x, m, rt, fs, fe, query_time=query
    )

    assert g.edge_mask.any()

    flow_index = g.edge_index[..., 1]
    expanded_end = fe[:, None, :].expand(B, N, Tf)
    active_end = torch.gather(expanded_end, 2, flow_index)

    assert torch.all(active_end[g.edge_mask] <= 0.04 + 1e-9)

    relation_masks = g.edge_mask.reshape(B, N, 4, Tr, Tf)
    assert not relation_masks[..., 2].any()


def test_independent_rgb_flow_lengths():
    B, N, Tr, Tf, D = 2, 3, 4, 7, 8

    x, m = make_inputs(Tr, Tf, B, N, D)

    rt = torch.tensor([
        [0.01, 0.03, 0.05, 0.07],
        [0.02, 0.04, 0.06, 0.08],
    ])

    fs = torch.arange(Tf).float().unsqueeze(0).expand(B, Tf) * 0.01
    fe = fs + 0.01

    g = PointIntervalGraphBuilder()(x, m, rt, fs, fe)

    expected_edges = 4 * Tr * Tf

    assert g.edge_index.shape == (B, N, expected_edges, 2)
    assert g.edge_attr.shape == (B, N, expected_edges, 5)
    assert g.edge_mask.shape == (B, N, expected_edges)
    assert g.edge_type.shape == (B, N, expected_edges)

    assert g.node_x["rgb_trend"].shape == (B, N, Tr, D)
    assert g.node_x["flow_trend"].shape == (B, N, Tf, D)

    assert set(g.edge_type.unique().tolist()) == {0, 1, 2, 3}


def test_padding_intervals_are_ignored():
    B, N, Tr, Tf, D = 1, 1, 3, 4, 4

    x, m = make_inputs(Tr, Tf, B, N, D)

    m["flow_trend"][:, :, 3] = False
    m["flow_event"][:, :, 3] = False

    rt = torch.tensor([[0.02, 0.04, 0.06]])
    fs = torch.tensor([[0.00, 0.02, 0.04, 0.00]])
    fe = torch.tensor([[0.02, 0.04, 0.06, 0.00]])

    g = PointIntervalGraphBuilder()(x, m, rt, fs, fe)

    relation_masks = g.edge_mask.reshape(B, N, 4, Tr, Tf)
    assert not relation_masks[..., 3].any()


def test_invalid_flow_interval_is_rejected():
    x, m = make_inputs(3, 3, B=1, N=1, D=4)

    fs = torch.tensor([[0.00, 0.02, 0.04]])
    fe = torch.tensor([[0.02, 0.04, 0.04]])

    with torch.no_grad():
        try:
            PointIntervalGraphBuilder()(
                x, m,
                torch.tensor([[0.00, 0.02, 0.04]]),
                fs, fe
            )
        except ValueError as exc:
            assert "positive duration" in str(exc)
        else:
            raise AssertionError("Invalid interval was not rejected")
