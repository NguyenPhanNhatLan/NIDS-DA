import unittest

import torch

from evaluation.v5e_gradient_conflict import gradient_summary, measure_conflict
from models.baseline import BaselineMLP
from models.hda_v1 import HDAV1Model
from models.hda_v5d import HDAV5DModel


class GradientConflictTests(unittest.TestCase):
    def test_known_opposition_and_zero_norm(self):
        vectors = {'hidden': torch.tensor([1., 0.]), 'normal': torch.zeros(2),
                   'rank': torch.zeros(2), 'attack': torch.tensor([-2., 0.])}
        result = gradient_summary(vectors)
        self.assertAlmostEqual(result['cosine_attack_vs_hidden_normal_rank'], -1.)
        self.assertAlmostEqual(result['attack_to_other_norm_ratio'], 2.)
        vectors['attack'] = torch.zeros(2)
        self.assertIsNone(gradient_summary(vectors)['cosine_attack_vs_hidden_normal_rank'])

    def test_diagnostic_leaves_all_model_state_unchanged(self):
        torch.manual_seed(42)
        source = BaselineMLP(4).eval()
        teacher = HDAV1Model(3, source).eval().requires_grad_(False)
        model = HDAV5DModel(source, teacher.adapter).train()
        for p in model.adapter.parameters():
            p.grad = torch.ones_like(p)
        sx, tx = torch.randn(8, 4), torch.randn(8, 3)
        hidden, latent = model.source_representations(sx)
        batch = {'source_hidden': hidden, 'target': tx,
                 'source_normal': latent[:4], 'source_attack': latent[4:],
                 'target_normal': tx[:4], 'target_attack': tx[4:],
                 'bandwidth_squared': {'hidden': 1., 'normal': 1., 'attack': 1.}}
        weights = {'hidden': 1., 'normal': .05, 'attack': .02, 'rank': .1}
        states = [{k: v.clone() for k, v in m.state_dict().items()} for m in (model, teacher)]
        modes = [[m.training for m in root.modules()] for root in (model, teacher)]
        flags = [p.requires_grad for p in model.parameters()]
        grads = [p.grad.clone() for p in model.adapter.parameters()]
        rng = torch.get_rng_state().clone()
        first = measure_conflict(model, teacher, batch, (.5, 1., 2.), weights)
        second = measure_conflict(model, teacher, batch, (.5, 1., 2.), weights)
        self.assertEqual(first, second)
        self.assertEqual(first['weights'], weights)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for root, state, mode in zip((model, teacher), states, modes):
            for key, value in root.state_dict().items():
                self.assertTrue(torch.equal(value, state[key]), key)
            self.assertEqual([m.training for m in root.modules()], mode)
        self.assertEqual([p.requires_grad for p in model.parameters()], flags)
        for p, before in zip(model.adapter.parameters(), grads):
            self.assertTrue(torch.equal(p.grad, before))
        no_attack = measure_conflict(model, teacher, batch, (.5, 1., 2.), {**weights, 'attack': 0.})
        self.assertEqual(no_attack['weighted_gradient_norms']['attack'], 0.)
        self.assertIsNone(no_attack['cosine_attack_vs_hidden_normal_rank'])


if __name__ == '__main__':
    unittest.main()
