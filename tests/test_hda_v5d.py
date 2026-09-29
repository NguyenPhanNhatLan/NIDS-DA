import unittest

import torch
from torch.nn import functional as F

from models.baseline import BaselineMLP
from models.hda_v1 import HDAV1Model
from models.hda_v5d import HDAV5DModel
from training.hda_v5d import baseline_class_weights, train_model


class V5dTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.source = BaselineMLP(4).eval()
        self.teacher = HDAV1Model(3, self.source).eval().requires_grad_(False)
        self.model = HDAV5DModel(self.source, self.teacher.adapter)

    def test_classifier_is_a_copy_not_shared_storage(self):
        before = self.source.classifier[0].weight.detach().clone()
        self.assertNotEqual(self.model.classifier[0].weight.data_ptr(),
                            self.source.classifier[0].weight.data_ptr())
        with torch.no_grad():
            self.model.classifier[0].weight[0, 0].add_(1.)
        self.assertTrue(torch.equal(before, self.source.classifier[0].weight))
        self.assertTrue(torch.equal(before, self.teacher.classifier[0].weight))
        for actual, expected in zip(self.model.adapter.parameters(), self.teacher.adapter.parameters()):
            self.assertTrue(torch.equal(actual, expected))
            self.assertNotEqual(actual.data_ptr(), expected.data_ptr())

    def test_source_ce_trains_only_private_classifier(self):
        self.model.train()
        frozen_before = {k: v.clone() for k, v in self.model.source_encoder.state_dict().items()}
        classifier_before = self.model.classifier[0].weight.detach().clone()
        _, logits = self.model.forward_source(torch.randn(8, 4))
        F.cross_entropy(logits, torch.tensor([0, 1] * 4), weight=baseline_class_weights([3, 7])).backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in self.model.classifier.parameters()))
        self.assertTrue(all(p.grad is None for p in self.model.source_encoder.parameters()))
        self.assertTrue(all(p.grad is None for p in self.model.adapter.parameters()))
        optimizer = torch.optim.Adam(self.model.optimizer_groups(1e-4, 1e-5))
        optimizer.step()
        self.assertFalse(torch.equal(classifier_before, self.model.classifier[0].weight))
        for k, v in self.model.source_encoder.state_dict().items():
            self.assertTrue(torch.equal(v, frozen_before[k]), k)

    def test_target_gradients_cross_frozen_shared_layers(self):
        self.model.train()
        _, logits = self.model.forward_target(torch.randn(8, 3))
        F.cross_entropy(logits, torch.tensor([0, 1] * 4)).backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in self.model.adapter.parameters()))
        self.assertTrue(all(p.grad is None for p in self.model.source_encoder.parameters()))

    def test_bn_policy_and_context_restore_on_exception(self):
        self.model.train()
        self.assertFalse(self.model.source_encoder.bn1.training)
        self.assertFalse(self.model.source_encoder.bn2.training)
        self.assertFalse(self.model.source_encoder.dropout.training)
        bn = self.model.adapter.layers[1]
        self.model.forward_target(torch.randn(8, 3))
        count = bn.num_batches_tracked.clone()
        with self.assertRaises(RuntimeError):
            with self.model.balanced_adapter_batch():
                self.assertFalse(bn.training)
                self.model.forward_target(torch.randn(8, 3))
                raise RuntimeError("test restore")
        self.assertTrue(bn.training)
        self.assertTrue(torch.equal(count, bn.num_batches_tracked))

    def test_optimizer_contains_exactly_adapter_and_classifier(self):
        optimizer = torch.optim.Adam(self.model.optimizer_groups(1e-4, 1e-5))
        actual = [id(p) for g in optimizer.param_groups for p in g["params"]]
        expected = {id(p) for p in self.model.adapter.parameters()} | {id(p) for p in self.model.classifier.parameters()}
        self.assertEqual(set(actual), expected)
        self.assertEqual(len(actual), len(expected))
        self.assertEqual([g["lr"] for g in optimizer.param_groups], [1e-4, 1e-5])

    def test_baseline_weighting(self):
        self.assertTrue(torch.allclose(baseline_class_weights([2, 8]), torch.tensor([2.5, 0.625])))
        for counts in ([0, 4], [-1, 4], [1], [1, float("nan")]):
            with self.assertRaises(ValueError):
                baseline_class_weights(counts)

    def test_training_step_preserves_teacher_source_and_frozen_buffers(self):
        teacher_before = {k: v.clone() for k, v in self.teacher.state_dict().items()}
        source_before = {k: v.clone() for k, v in self.source.state_dict().items()}
        frozen_before = {k: v.clone() for k, v in self.model.source_encoder.state_dict().items()}
        adapter_before = self.model.adapter.layers[0].weight.detach().clone()
        classifier_before = self.model.classifier[0].weight.detach().clone()
        x, y, target = torch.randn(8, 4), torch.tensor([0, 1] * 4), torch.randn(8, 3)
        _, latent = self.model.source_representations(x)
        config = {"adapter_lr": 1e-4, "classifier_lr": 1e-5, "weight_decay": 1e-4,
                  "loss_weights": {"hidden": 1., "normal": .05, "attack": .02, "rank": .1, "source": .1}}
        history = train_model(self.model, self.teacher, [(x, y)], [target],
                              {0: latent[y == 0], 1: latent[y == 1]},
                              {0: target[:4], 1: target[4:]}, baseline_class_weights([4, 4]),
                              config, {"class_batch_size": 4, "epochs": 1})
        for model, before in ((self.teacher, teacher_before), (self.source, source_before),
                              (self.model.source_encoder, frozen_before)):
            for k, v in model.state_dict().items():
                self.assertTrue(torch.equal(v, before[k]), k)
        self.assertFalse(torch.equal(adapter_before, self.model.adapter.layers[0].weight))
        self.assertFalse(torch.equal(classifier_before, self.model.classifier[0].weight))
        self.assertEqual(self.model.adapter.layers[1].num_batches_tracked.item(), 1)
        self.assertTrue(all(p.grad is None for p in self.teacher.parameters()))
        row = history[0]
        expected_loss = sum(config["loss_weights"][k] * row[k] for k in config["loss_weights"])
        self.assertAlmostEqual(row["loss"], expected_loss, places=5)


if __name__ == "__main__":
    unittest.main()
