import contextlib
import io
import unittest

import torch

from models.baseline import BaselineMLP
from models.hda_v1 import HDAV1Model
from models.hda_residual import ResidualHDAModel, ResidualTargetCorrection
from training.baseline import set_seed
from training.hda_residual import ranking_loss, train_residual


class ResidualHDATests(unittest.TestCase):
    def make_models(self):
        set_seed(42)
        source = BaselineMLP(4).eval()
        teacher = HDAV1Model(3, source).eval()
        student = ResidualHDAModel(source, teacher.adapter)
        return teacher, student

    def test_zero_residual_matches_v2_and_frozen_modes(self):
        teacher, model = self.make_models()
        features = torch.randn(16, 3)
        expected = teacher(features)
        model.train()
        actual = model(features)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertFalse(model.source.training)
        self.assertFalse(model.adapter.training)
        self.assertTrue(model.correction.training)
        for name, parameter in model.named_parameters():
            self.assertEqual(parameter.requires_grad, name.startswith("correction."))

    def test_zero_final_layer_first_gradient(self):
        correction = ResidualTargetCorrection()
        self.assertEqual(correction.alpha.item(), 1.0)
        self.assertEqual(correction.block[-1].weight.count_nonzero().item(), 0)
        self.assertEqual(correction.block[-1].bias.count_nonzero().item(), 0)
        correction(torch.randn(8, 256)).square().sum().backward()
        self.assertEqual(correction.alpha.grad.item(), 0)
        self.assertGreater(correction.block[-1].weight.grad.abs().sum().item(), 0)
        self.assertGreater(correction.block[-1].bias.grad.abs().sum().item(), 0)
        for parameter in correction.block[:-1].parameters():
            self.assertEqual(parameter.grad.abs().sum().item(), 0)

    def test_training_changes_only_correction(self):
        for rank_weight in (0.0, 0.01):
            with self.subTest(rank_weight=rank_weight):
                _, model = self.make_models()
                frozen = {key: value.clone() for key, value in model.state_dict().items()
                          if not key.startswith("correction.")}
                block_before = {key: value.clone() for key, value in model.correction.block.state_dict().items()}
                source_loader = [(torch.randn(8, 4), torch.tensor([0, 1] * 4))]
                target_loader = [torch.randn(8, 3) for _ in range(3)]
                source_pools = {label: torch.randn(12, 168) for label in (0, 1)}
                target_pools = {label: torch.randn(12, 3) for label in (0, 1)}
                weights = {"hidden": 1.0, "normal": 0.05,
                           "attack": 0.02 if rank_weight else 0.05, "rank": rank_weight}
                with contextlib.redirect_stdout(io.StringIO()):
                    history = train_residual(model, source_loader, target_loader,
                                             source_pools, target_pools, weights,
                                             epochs=1, class_batch_size=4)
                for key, value in frozen.items():
                    torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
                self.assertNotEqual(model.correction.alpha.item(), 0)
                self.assertTrue(any(not torch.equal(value, model.correction.block.state_dict()[key])
                                    for key, value in block_before.items()))
                row = history[0]
                self.assertAlmostEqual(row["loss"], sum(weights[key] * row[key] for key in weights), places=6)

    def test_ranking_loss(self):
        teacher = torch.tensor([-2., -1., 1., 2.], requires_grad=True)
        self.assertAlmostEqual(ranking_loss(teacher, 3 * teacher.detach() + 5).item(), 0, places=6)
        self.assertAlmostEqual(ranking_loss(teacher, -teacher.detach()).item(), 2, places=6)
        student = torch.tensor([1., 0., 2., 3.], requires_grad=True)
        ranking_loss(teacher, student).backward()
        self.assertIsNone(teacher.grad)
        self.assertTrue(torch.isfinite(student.grad).all())
        constant = torch.ones(4, requires_grad=True)
        loss = ranking_loss(teacher, constant)
        loss.backward()
        self.assertTrue(torch.isfinite(constant.grad).all())
        self.assertEqual(ranking_loss(torch.ones(4), student).item(), 0)


if __name__ == "__main__":
    unittest.main()
