import contextlib
import io
import unittest
from unittest.mock import patch

import torch

from models.baseline import BaselineMLP
from models.hda_v1 import HDAV1Model
from training.baseline import set_seed
from training.hda_v4 import train_hda_v4
from training.hda_v5b import ranking_loss, train_hda_v5b


class V5bTests(unittest.TestCase):
    def inputs(self):
        set_seed(42)
        source = BaselineMLP(4)
        teacher = HDAV1Model(3, source)
        source_loader = [(torch.randn(8, 4), torch.tensor([0, 1] * 4))]
        target_loader = [torch.randn(8, 3) for _ in range(3)]
        source_pools = {label: torch.randn(12, 168) for label in (0, 1)}
        target_pools = {label: torch.randn(12, 3) for label in (0, 1)}
        return source, teacher, source_loader, target_loader, source_pools, target_pools

    def test_adapter_only_and_bn_updates(self):
        for attack_weight in (0.05, 0.02):
            with self.subTest(attack=attack_weight):
                inputs = self.inputs()
                source, teacher = inputs[:2]
                source_before = {key: value.clone() for key, value in source.state_dict().items()}
                teacher_before = {key: value.clone() for key, value in teacher.state_dict().items()}
                adapter_before = {key: value.clone() for key, value in teacher.adapter.state_dict().items()}
                def create_student(*args):
                    student = HDAV1Model(*args)
                    original_load = student.adapter.load_state_dict
                    def load(state):
                        for key in state:
                            torch.testing.assert_close(state[key], adapter_before[key], rtol=0, atol=0)
                        return original_load(state)
                    student.adapter.load_state_dict = load
                    return student
                weights = {"hidden": 1., "normal": 0.05, "attack": attack_weight, "rank": 0.10}
                with patch("training.hda_v5b.HDAV1Model", side_effect=create_student), \
                        patch("training.hda_v5b.torch.optim.Adam", wraps=torch.optim.Adam) as adam, \
                        contextlib.redirect_stdout(io.StringIO()):
                    student, history = train_hda_v5b(*inputs, weights, epochs=1, class_batch_size=4)
                self.assertIsInstance(student, HDAV1Model)
                self.assertFalse(hasattr(student, "correction"))
                self.assertEqual(adam.call_args.kwargs, {"lr": 0.001, "weight_decay": 0.0001})
                for name, parameter in student.named_parameters():
                    self.assertEqual(parameter.requires_grad, name.startswith("encoder.adapter."))
                for key, value in source_before.items():
                    torch.testing.assert_close(source.state_dict()[key], value, rtol=0, atol=0)
                for key, value in teacher_before.items():
                    torch.testing.assert_close(teacher.state_dict()[key], value, rtol=0, atol=0)
                for key, value in adapter_before.items():
                    if key.endswith("num_batches_tracked"):
                        self.assertEqual(student.adapter.state_dict()[key].item(), value.item() + 3)
                self.assertTrue(any(not torch.equal(value, student.adapter.state_dict()[key])
                                    for key, value in adapter_before.items() if key.endswith("weight")))
                row = history[0]
                self.assertAlmostEqual(row["loss"], sum(weights[key] * row[key] for key in weights), places=6)

    def test_zero_rank_matches_v5a(self):
        inputs = self.inputs()
        set_seed(123)
        with contextlib.redirect_stdout(io.StringIO()):
            expected, _ = train_hda_v4(*inputs, epochs=1, class_batch_size=4, lambda_conditional=0.10)
        inputs = self.inputs()
        set_seed(123)
        weights = {"hidden": 1., "normal": 0.05, "attack": 0.05, "rank": 0.0}
        with contextlib.redirect_stdout(io.StringIO()):
            actual, _ = train_hda_v5b(*inputs, weights, epochs=1, class_batch_size=4)
        for key, value in expected.state_dict().items():
            torch.testing.assert_close(actual.state_dict()[key], value, atol=2e-6, rtol=1e-5)

    def test_rank_teacher_is_detached(self):
        teacher = torch.tensor([-2., -1., 1., 2.], requires_grad=True)
        student = torch.tensor([-1., 0., 3., 2.], requires_grad=True)
        ranking_loss(teacher, student).backward()
        self.assertIsNone(teacher.grad)
        self.assertTrue(torch.isfinite(student.grad).all())
        self.assertGreater(student.grad.abs().sum().item(), 0)
        self.assertEqual(ranking_loss(torch.ones(4), student).item(), 0)


if __name__ == "__main__":
    unittest.main()
