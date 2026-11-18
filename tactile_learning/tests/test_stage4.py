import torch
from tactile.models.online_state import CausalStreamBuffer
from tactile.losses.sampling_consistency import sampling_consistency_loss
from tactile.evaluation.early_recognition import earliest_confident_time
from tactile.data_sampling import subsample_by_period

def test_flow_only_after_interval_end():
    b=CausalStreamBuffer()
    b.push_flow(torch.ones(3), 0.0, 0.1)
    assert bool(b.available_flow_mask([0.0],[0.1],0.099)[0]) is False
    assert bool(b.available_flow_mask([0.0],[0.1],0.1)[0]) is True

def test_sampling_is_timestamp_based():
    t=torch.tensor([0., .01, .02, .033, .05, .067])
    idx=subsample_by_period(t, 1/30)
    assert idx.tolist()==[0,3,5]

def test_consistency_finite():
    a=torch.randn(4,8,requires_grad=True); b=torch.randn(4,8,requires_grad=True)
    loss=sampling_consistency_loss(a,b)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all()

def test_early_recognition():
    logits=torch.tensor([[2.,0.],[3.,0.],[4.,0.],[4.,0.]])
    t=torch.tensor([0.,.1,.2,.3])
    assert abs(earliest_confident_time(logits,t,.8,2) - 0.1) < 1e-6
