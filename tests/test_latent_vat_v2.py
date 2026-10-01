import unittest

import torch
from torch import nn

from training.latent_vat_v2 import latent_vat_loss


class LatentVATTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.classifier = nn.Sequential(
            nn.Linear(6, 4),
            nn.ReLU(),
            nn.Linear(4, 2),
        ).double()
        self.classifier.eval()
        for p in self.classifier.parameters():
            p.requires_grad = False

    def test_finite_and_adapter_gradient(self):
        z = torch.randn(8, 6, dtype=torch.float64, requires_grad=True)
        loss = latent_vat_loss(
            z, self.classifier,
            epsilon_ratio=0.05,
            xi_ratio=1e-3,
            power_iterations=1,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(loss.item(), -1e-12)
        grad = torch.autograd.grad(loss, z)[0]
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(torch.linalg.vector_norm(grad).item(), 0.0)

    def test_zero_epsilon_is_zero_consistency(self):
        z = torch.randn(8, 6, dtype=torch.float64, requires_grad=True)
        loss = latent_vat_loss(
            z, self.classifier,
            epsilon_ratio=0.0,
            xi_ratio=1e-3,
            power_iterations=1,
        )
        self.assertLess(abs(loss.item()), 1e-10)

    def test_classifier_stays_frozen(self):
        z = torch.randn(8, 6, dtype=torch.float64, requires_grad=True)
        loss = latent_vat_loss(z, self.classifier)
        loss.backward()
        self.assertTrue(all(p.grad is None for p in self.classifier.parameters()))

    def test_invalid_parameters(self):
        z = torch.randn(8, 6)
        with self.assertRaises(ValueError):
            latent_vat_loss(z, self.classifier, epsilon_ratio=-0.1)
        with self.assertRaises(ValueError):
            latent_vat_loss(z, self.classifier, xi_ratio=0.0)
        with self.assertRaises(ValueError):
            latent_vat_loss(z, self.classifier, power_iterations=0)


if __name__ == "__main__":
    unittest.main()
