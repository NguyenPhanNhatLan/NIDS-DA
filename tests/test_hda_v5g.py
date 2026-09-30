import unittest
from unittest.mock import patch

import numpy as np
import torch
from scipy.stats import rankdata

from models.baseline import BaselineMLP
from models.hda_v1 import HDAV1Model
from models.hda_v5d import HDAV5DModel
from training.hda_v5g import (average_percentile_rank, build_geometry_middle_pool,
                               component_weights, train_model, V5F_WEIGHTS)


class ToyV2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.))

    def forward(self, x):
        return x * self.scale, torch.cat((torch.zeros_like(x), x * self.scale), 1)


class V5gTests(unittest.TestCase):
    def test_geometry_ranks_and_preserves_original_model(self):
        values = np.array([2., 1., 1., 5.])
        np.testing.assert_allclose(average_percentile_rank(values), (rankdata(values)-1)/3)
        model = ToyV2().train()
        before = model.scale.detach().clone()
        with patch('training.hda_v5g.make_teacher_loader', return_value=[torch.arange(6.).reshape(-1,1)]):
            pool = build_geometry_middle_pool(model, 'unused', 1, 6,
                       torch.tensor([0.]), torch.tensor([5.]), 0., 5., 2.)
        self.assertTrue(model.training)
        self.assertTrue(model.scale.requires_grad)
        torch.testing.assert_close(model.scale, before)
        self.assertEqual(pool['metadata']['rows_scored'], 6)
        torch.testing.assert_close(pool['features'].flatten(), torch.arange(1.,5.))
        torch.testing.assert_close(pool['attack_weight'], torch.linspace(0,1,4).square())
        w = component_weights(torch.tensor([1.,3.]), 2, .5, 'cpu', torch.float32)
        torch.testing.assert_close(w, torch.tensor([.25,.25,.125,.375]))

    def test_one_step_cpu(self):
        self.run_step('cpu')

    @unittest.skipUnless(torch.backends.mps.is_available(), 'MPS unavailable')
    def test_one_step_mps(self):
        self.run_step('mps')

    def run_step(self, device):
        torch.manual_seed(42)
        source = BaselineMLP(4).eval()
        teacher = HDAV1Model(3, source).eval().requires_grad_(False).to(device)
        student = HDAV5DModel(source, teacher.adapter).to(device)
        s, t = torch.randn(8,4), torch.randn(8,3)
        _, z = student.source_representations(s.to(device))
        pools = {0:z[:4].cpu(), 1:z[4:].cpu()}
        middle = {'features': t, 'normal_weight': torch.linspace(.1,.9,8),
                  'attack_weight': torch.linspace(.9,.1,8)}
        frozen = {k:v.clone() for k,v in teacher.state_dict().items()}
        config = dict(adapter_lr=1e-4, classifier_lr=1e-5, weight_decay=1e-4,
                      loss_weights=V5F_WEIGHTS, kernel_scales=[.25,.5,1.,2.,4.],
                      conditional_anchor_mass=.5)
        rows = train_model(student, teacher, [(s,torch.tensor([0,1]*4))], [t],
                           pools, {0:t[:4],1:t[4:]}, middle, torch.ones(2), config,
                           {'epochs':1,'class_batch_size':4})
        self.assertEqual(rows[0]['steps'], 1)
        self.assertAlmostEqual(rows[0]['loss'], sum(V5F_WEIGHTS[k]*rows[0][k] for k in V5F_WEIGHTS), places=5)
        self.assertEqual(student.adapter.layers[1].num_batches_tracked.item(), 1)
        for k,v in teacher.state_dict().items():
            torch.testing.assert_close(v, frozen[k], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
