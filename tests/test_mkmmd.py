import unittest

import torch

from training.adaptation import mmd_loss
from training.mkmmd import mk_mmd_loss


class MKMMDTests(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(42)
        self.s = torch.randn(7, 3, generator=generator, dtype=torch.float64)
        self.t = torch.randn(9, 3, generator=generator, dtype=torch.float64) + .5

    def test_single_scale_matches_frozen_rbf_including_unequal_batches(self):
        expected, sigma = mmd_loss(self.s, self.t)
        actual, actual_sigma = mk_mmd_loss(self.s, self.t, scales=(1.,))
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual_sigma, sigma)

    def test_mixture_matches_manual_kernel_formula(self):
        s, t = self.s, self.t[:len(self.s)]
        scales = (.25, 1., 4.)
        def kernel(x, y):
            d = ((x[:, None] - y[None, :]) ** 2).sum(-1)
            return sum(torch.exp(-d / (2 * 1.3 * scale)) for scale in scales) / len(scales)
        expected = kernel(s, s).mean() + kernel(t, t).mean() - 2 * kernel(s, t).mean()
        actual, _ = mk_mmd_loss(s, t, scales=scales, bandwidth_squared=1.3)
        torch.testing.assert_close(actual, expected)

    def test_symmetry_identity_and_collapsed_samples(self):
        loss, _ = mk_mmd_loss(self.s, self.t)
        reverse, _ = mk_mmd_loss(self.t, self.s)
        torch.testing.assert_close(loss, reverse)
        same, _ = mk_mmd_loss(self.s, self.s)
        self.assertAlmostEqual(same.item(), 0., places=10)
        collapsed = torch.zeros(4, 3, dtype=torch.float64, requires_grad=True)
        zero, sigma = mk_mmd_loss(collapsed, collapsed)
        zero.backward()
        self.assertEqual(sigma.item(), 1.)
        self.assertTrue(torch.isfinite(collapsed.grad).all())
        self.assertGreaterEqual(loss.item(), -1e-10)

    def test_gradient_with_fixed_detached_bandwidth(self):
        s = self.s[:3].clone().requires_grad_()
        t = self.t[:3].clone().requires_grad_()
        base = torch.tensor(1.3, dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(
            lambda x, y: mk_mmd_loss(x, y, bandwidth_squared=base)[0], (s, t)))
        loss, sigma = mk_mmd_loss(s, t, bandwidth_squared=base)
        loss.backward()
        self.assertIsNone(base.grad)
        self.assertFalse(sigma.requires_grad)
        self.assertTrue(torch.isfinite(t.grad).all())
        self.assertGreater(t.grad.abs().sum().item(), 0.)

    def test_invalid_inputs(self):
        for scales in ((), (0.,), (-1.,), (float('nan'),)):
            with self.assertRaises(ValueError):
                mk_mmd_loss(self.s, self.t, scales=scales)
        for bandwidth in (0., -1., float('inf')):
            with self.assertRaises(ValueError):
                mk_mmd_loss(self.s, self.t, bandwidth_squared=bandwidth)
        for s, t in ((self.s[:0], self.t), (self.s, self.t[:, :2]),
                     (self.s, self.t.float()), (self.s * float('nan'), self.t)):
            with self.assertRaises(ValueError):
                mk_mmd_loss(s, t)


if __name__ == '__main__':
    unittest.main()
