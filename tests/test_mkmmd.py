import unittest
from unittest.mock import patch

import torch

from training.adaptation import mmd_loss
from training.mkmmd import mk_mmd_loss, MKMMDLoss
from training.v5e_performance import legacy_mk_mmd_loss


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

    def test_one_cdist_and_one_exp(self):
        with (patch("torch.cdist", wraps=torch.cdist) as distances,
              patch("torch.exp", wraps=torch.exp) as exponential):
            mk_mmd_loss(self.s, self.t)
        self.assertEqual(distances.call_count, 1)
        self.assertEqual(exponential.call_count, 1)

    def compare_legacy(self, device, dtype, atol, rtol):
        generator = torch.Generator().manual_seed(9)
        s = torch.randn(40, 8, generator=generator, dtype=dtype).to(device)
        t = torch.randn(43, 8, generator=generator, dtype=dtype).to(device)
        scales = (.25, .5, 1., 2., 4.)
        for base in (None, 2.3):
            a, b = s.clone().requires_grad_(), t.clone().requires_grad_()
            old, old_sigma = legacy_mk_mmd_loss(a, b, scales, bandwidth_squared=base)
            old_grad = torch.autograd.grad(old, (a, b))
            c, d = s.clone().requires_grad_(), t.clone().requires_grad_()
            new, new_sigma = mk_mmd_loss(c, d, scales, bandwidth_squared=base)
            new_grad = torch.autograd.grad(new, (c, d))
            torch.testing.assert_close(new, old, atol=atol, rtol=rtol)
            torch.testing.assert_close(new_sigma, old_sigma, atol=atol, rtol=rtol)
            for actual, expected in zip(new_grad, old_grad):
                torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)

    def test_legacy_loss_and_gradient_cpu(self):
        self.compare_legacy("cpu", torch.float64, 1e-9, 1e-6)

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS unavailable")
    def test_legacy_loss_and_gradient_mps(self):
        self.compare_legacy("mps", torch.float32, 2e-5, 2e-3)

    def test_cached_scales_and_debug_check(self):
        kernel = MKMMDLoss().to(dtype=torch.float64)
        result, _ = kernel(self.s, self.t)
        expected, _ = mk_mmd_loss(self.s, self.t)
        torch.testing.assert_close(result, expected)
        with self.assertRaises(ValueError):
            kernel(self.s * float('nan'), self.t, validate=True)


if __name__ == '__main__':
    unittest.main()
