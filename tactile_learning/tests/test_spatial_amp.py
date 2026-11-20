import pytest
import torch

from tactile.models.spatial_graph import TactileSpatialGraphBranch


def _make_inputs(device):
    # The second sample has no valid observations. It exercises the
    # all-masked temporal-attention row while the first sample remains valid.
    rgb = torch.randn(2, 1, 3, 3, 32, 32, device=device)
    mask = torch.tensor(
        [[[True, True, True]], [[False, False, False]]],
        dtype=torch.bool,
        device=device,
    )
    return rgb, mask


def _run_forward_backward(device, use_amp):
    torch.manual_seed(17)
    model = TactileSpatialGraphBranch(
        feature_dim=32,
        spatial_dim=32,
        dropout=0.0,
        graph_layers=1,
        adaptive_neighbors=4,
    ).to(device)
    rgb, mask = _make_inputs(device)
    rgb.requires_grad_(True)

    amp_context = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if use_amp
        else torch.autocast(device_type=device.type, enabled=False)
    )
    with amp_context:
        episode, per_sensor = model(rgb, mask)

    assert episode.shape == (2, 32)
    assert per_sensor.shape == (2, 1, 32)
    assert torch.isfinite(episode).all()
    assert torch.isfinite(per_sensor).all()
    assert torch.count_nonzero(episode[1]) == 0

    loss = episode.float().square().mean() + per_sensor.float().square().mean()
    loss.backward()
    assert rgb.grad is not None
    assert torch.isfinite(rgb.grad).all()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)


def test_spatial_temporal_attention_fp32_forward_backward():
    _run_forward_backward(torch.device("cpu"), use_amp=False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_spatial_temporal_attention_cuda_amp_fp16_forward_backward():
    _run_forward_backward(torch.device("cuda"), use_amp=True)
