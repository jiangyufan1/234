import unittest
import torch
from tactile.models.trend_event import CausalTrendEventDecomposer


class TrendEventTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)

    def test_shapes_and_backward(self):
        m = CausalTrendEventDecomposer(12, num_atoms=4)
        x = torch.randn(2, 3, 9, 12, requires_grad=True)
        times = (torch.arange(9, dtype=torch.float32)[None, :] * .01).expand(2, -1)
        mask = torch.ones(2, 3, 9, dtype=torch.bool)
        y = m(x, times, mask)
        self.assertEqual(y.trend.shape, x.shape)
        self.assertEqual(y.gates.shape, (2, 3, 9, 4))
        (y.event.square().mean() + m.total_auxiliary_loss(y)).backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_causal_past_unchanged(self):
        m = CausalTrendEventDecomposer(8, num_atoms=3).eval()
        x = torch.randn(1, 1, 10, 8)
        x2 = x.clone(); x2[:, :, 7:] += 100
        times = torch.arange(10, dtype=torch.float32)[None, :] * .01
        mask = torch.ones(1, 1, 10, dtype=torch.bool)
        with torch.no_grad():
            a, b = m(x, times, mask), m(x2, times, mask)
        torch.testing.assert_close(a.trend[:, :, :7], b.trend[:, :, :7])
        torch.testing.assert_close(a.event[:, :, :7], b.event[:, :, :7])

    def test_irregular_sampling_and_mask(self):
        m = CausalTrendEventDecomposer(6, num_atoms=3).eval()
        x = torch.randn(1, 2, 5, 6)
        times = torch.tensor([[0., .01, .03, .08, .10]])
        mask = torch.tensor([[[1,1,0,1,1],[1,1,1,1,1]]], dtype=torch.bool)
        x[0,0,2] = float("nan")
        with torch.no_grad(): y = m(x, times, mask)
        self.assertTrue(torch.isfinite(y.trend).all())
        self.assertTrue(torch.isfinite(y.event).all())
        self.assertTrue(torch.equal(y.trend[0,0,2], torch.zeros(6)))

    def test_constant_signal_has_zero_residual_after_first_observation(self):
        m = CausalTrendEventDecomposer(4, num_atoms=3).eval()
        x = torch.ones(1,1,12,4)*2.5
        times = torch.arange(12, dtype=torch.float32)[None,:]*.01
        mask = torch.ones(1,1,12,dtype=torch.bool)
        with torch.no_grad(): y=m(x,times,mask)
        self.assertLess(y.residual[:, :, 1:].abs().max().item(), 1e-6)

    def test_bad_time_order_fails(self):
        m=CausalTrendEventDecomposer(4, num_atoms=2)
        x=torch.randn(1,1,3,4)
        times=torch.tensor([[0.,.02,.01]])
        mask=torch.ones(1,1,3,dtype=torch.bool)
        with self.assertRaisesRegex(ValueError,"strictly increasing"):
            m(x,times,mask)


if __name__ == "__main__":
    unittest.main()
