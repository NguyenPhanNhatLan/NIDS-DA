import unittest

import torch

from training.mkmmd import mk_mmd_loss
from training.weighted_mkmmd import weighted_mk_mmd_loss


class WeightedMKMMDTests(unittest.TestCase):
    def test_uniform_matches_existing_kernel_and_gradients(self):
        torch.manual_seed(42)
        s = torch.randn(5, 3, dtype=torch.float64)
        t = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
        actual, _ = weighted_mk_mmd_loss(s, t, torch.ones(5), bandwidth_squared=1.3)
        expected, _ = mk_mmd_loss(s, t, bandwidth_squared=1.3)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(torch.autograd.grad(actual, t)[0],
                                   torch.autograd.grad(expected, t)[0])

    def test_manual_unequal_weighted_formula(self):
        s = torch.tensor([[0.], [1.]], dtype=torch.float64)
        t = torch.tensor([[2.], [3.], [4.]], dtype=torch.float64, requires_grad=True)
        ws = torch.tensor([.25, .75], dtype=torch.float64)
        wt = torch.tensor([.1, .3, .6], dtype=torch.float64)
        kernel = lambda a, b: torch.exp(-torch.cdist(a, b).square() / 2)
        expected = ws @ kernel(s, s) @ ws + wt @ kernel(t, t) @ wt - 2 * ws @ kernel(s, t) @ wt
        actual, _ = weighted_mk_mmd_loss(s, t, wt * 10, scales=[1.],
                                        source_weights=ws * 2, bandwidth_squared=1.)
        torch.testing.assert_close(actual, expected)
        self.assertTrue(torch.autograd.gradcheck(
            lambda x: weighted_mk_mmd_loss(s, x, wt, scales=[1.],
                                          source_weights=ws, bandwidth_squared=1.)[0], (t,)))

    def test_invalid_weights_rejected(self):
        x = torch.randn(3, 2)
        for weights in (torch.zeros(3), torch.tensor([-1., 1., 1.]),
                        torch.tensor([float('nan'), 1., 1.]), torch.ones(2)):
            with self.subTest(weights=weights):
                with self.assertRaises(ValueError):
                    weighted_mk_mmd_loss(x, x, weights)


if __name__ == '__main__':
    unittest.main()
