import unittest
from unittest.mock import patch

import torch

from models.baseline import BaselineMLP
from models.hda_v1 import HDAV1Model
from models.hda_v5d import HDAV5DModel
from training.hda_v5e import diagnostic_metrics, train_model
from training.mkmmd import mk_mmd_loss


class V5eTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.source = BaselineMLP(4).eval()
        self.teacher = HDAV1Model(3, self.source).eval().requires_grad_(False)
        self.student = HDAV5DModel(self.source, self.teacher.adapter)
        self.s, self.t = torch.randn(8, 4), torch.randn(8, 3)
        h, z = self.student.source_representations(self.s)
        self.pools = {0: z[:4], 1: z[4:]}
        self.batch = {"source_hidden": h, "target": self.t,
                      "source_normal": z[:4], "source_attack": z[4:],
                      "target_normal": self.t[:4], "target_attack": self.t[4:],
                      "bandwidth_squared": {"hidden": 1., "normal": 1., "attack": 1.}}

    def test_fixed_diagnostic_preserves_rng_bn_and_training_modes(self):
        self.student.train()
        before = {k: v.clone() for k, v in self.student.state_dict().items()}
        rng = torch.get_rng_state().clone()
        modes = [m.training for m in self.student.modules()]
        first = diagnostic_metrics(self.student, self.batch, (.5, 1., 2.))
        second = diagnostic_metrics(self.student, self.batch, (.5, 1., 2.))
        self.assertEqual(first, second)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(modes, [m.training for m in self.student.modules()])
        for key, value in self.student.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]), key)

    def test_three_mk_calls_preserve_ce_ranking_and_freezes(self):
        config = {"adapter_lr": 1e-4, "classifier_lr": 1e-5, "weight_decay": 1e-4,
                  "kernel_scales": [.25, .5, 1., 2., 4.],
                  "loss_weights": {"hidden": 1., "normal": .05, "attack": .02, "rank": .1, "source": .1}}
        teacher_before = {k: v.clone() for k, v in self.teacher.state_dict().items()}
        frozen_before = {k: v.clone() for k, v in self.student.source_encoder.state_dict().items()}
        with patch("training.hda_v5e.mk_mmd_loss", wraps=mk_mmd_loss) as kernel:
            rows = train_model(self.student, self.teacher,
                               [(self.s, torch.tensor([0, 1] * 4))], [self.t],
                               self.pools, {0: self.t[:4], 1: self.t[4:]}, torch.ones(2),
                               config, {"epochs": 1, "class_batch_size": 4})
        self.assertEqual(kernel.call_count, 3)
        self.assertTrue(all(c.kwargs["scales"] == config["kernel_scales"] for c in kernel.call_args_list))
        self.assertGreater(rows[0]["source"], 0.)
        self.assertAlmostEqual(rows[0]["loss"], sum(config["loss_weights"][k] * rows[0][k]
                                                  for k in config["loss_weights"]), places=5)
        for model, before in ((self.teacher, teacher_before), (self.student.source_encoder, frozen_before)):
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, before[key]), key)
        self.assertEqual(self.student.adapter.layers[1].num_batches_tracked.item(), 1)


if __name__ == '__main__':
    unittest.main()
