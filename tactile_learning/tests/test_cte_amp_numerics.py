import pytest
import torch

from tactile.models.cte_hgnn import (
    BidirectionalRelationMessage,
    masked_softmax,
)


def test_masked_softmax_fp16_all_invalid_rows_are_finite_and_zero():
    scores = torch.tensor(
        [[1.0, -2.0, 3.0], [4.0, 5.0, 6.0]],
        dtype=torch.float16,
        requires_grad=True,
    )
    mask = torch.tensor(
        [[True, False, True], [False, False, False]],
        dtype=torch.bool,
    )

    weights = masked_softmax(scores, mask, dim=-1)

    assert weights.dtype == torch.float16
    assert torch.isfinite(weights).all()
    assert torch.allclose(weights[1], torch.zeros_like(weights[1]))
    assert torch.allclose(weights[0].sum(), torch.tensor(1.0, dtype=torch.float16))

    weights.float().square().sum().backward()
    assert scores.grad is not None
    assert torch.isfinite(scores.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_relation_message_amp_zero_compatibility_has_finite_gradients():
    torch.manual_seed(2026)
    device = torch.device("cuda")
    dim, batch, nodes, tr, tf = 16, 2, 1, 3, 4

    module = BidirectionalRelationMessage(dim, edge_dim=5, dropout=0.0).to(device)
    rgb = torch.randn(batch, nodes, tr, dim, device=device, requires_grad=True)
    flow = torch.randn(batch, nodes, tf, dim, device=device, requires_grad=True)
    edge_attr = torch.randn(
        batch, nodes, tr, tf, 5, device=device, requires_grad=True
    )
    # Simulate compatibility underflowing to zero in half precision.
    with torch.no_grad():
        edge_attr[..., 4].zero_()
    edge_mask = torch.ones(batch, nodes, tr, tf, dtype=torch.bool, device=device)
    relation = torch.randn(dim, device=device, requires_grad=True)

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        rgb_msg, flow_msg, _, _ = module(
            rgb, flow, edge_attr, edge_mask, relation
        )
        loss = rgb_msg.float().square().mean() + flow_msg.float().square().mean()

    assert torch.isfinite(rgb_msg).all()
    assert torch.isfinite(flow_msg).all()
    assert torch.isfinite(loss)
    loss.backward()

    tensors = [rgb, flow, edge_attr, relation]
    tensors += [p for p in module.parameters()]
    for tensor in tensors:
        if tensor.grad is not None:
            assert torch.isfinite(tensor.grad).all(), (
                f"Non-finite gradient in tensor shape {tuple(tensor.shape)}"
            )
