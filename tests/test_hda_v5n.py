"""V5n EMA and BN invariants. Tests are provided but not run during authoring."""
import copy
import unittest
from unittest.mock import patch

import torch
from torch import nn

from models.baseline import BaselineMLP
from models.hda_v1 import HDAV1Model
from models.hda_v5d import HDAV5DModel
from training.hda_v5d import train_model as train_v5d
from training.hda_v5n import V5D_WEIGHTS, apply_ema, recalibrate_adapter_bn, train_model, update_ema


class V5nTests(unittest.TestCase):
    def test_ema_parameter_update(self):
        layer = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            layer.weight.fill_(2.)
        ema = {name: p.detach().clone() for name, p in layer.named_parameters()}
        with torch.no_grad():
            layer.weight.fill_(4.)
        update_ema(layer, ema, .999)
        torch.testing.assert_close(ema['weight'], torch.full_like(layer.weight, 2.002))
        torch.testing.assert_close(layer.weight, torch.full_like(layer.weight, 4.))
        apply_ema(layer, ema)
        torch.testing.assert_close(layer.weight, ema['weight'])

    def test_bn_recalibration_changes_buffers_not_parameters(self):
        adapter = nn.Sequential(nn.Linear(2, 2, bias=False), nn.BatchNorm1d(2), nn.ReLU()).eval()
        with torch.no_grad():
            adapter[0].weight.copy_(torch.eye(2))
            adapter[1].running_mean.fill_(999.)
        before = {k:p.clone() for k,p in adapter.named_parameters()}
        batch = torch.tensor([[1., 2.], [3., 6.]])
        info = recalibrate_adapter_bn(adapter, [batch, batch], torch.device('cpu'))
        torch.testing.assert_close(adapter[1].running_mean, torch.tensor([2., 4.]))
        torch.testing.assert_close(adapter[1].running_var, torch.tensor([2., 8.]))
        self.assertEqual(info['rows'], 4)
        self.assertEqual(adapter[1].momentum, .1)
        self.assertFalse(adapter.training)
        self.assertFalse(adapter[1].training)
        for k,p in adapter.named_parameters():
            torch.testing.assert_close(p, before[k], rtol=0, atol=0)
        with self.assertRaises(ValueError):
            recalibrate_adapter_bn(adapter, [], torch.device('cpu'))
        self.assertEqual(adapter[1].momentum, .1)

    def test_joint_step_matches_v5d_before_ema_application(self):
        torch.manual_seed(42)
        source = BaselineMLP(4).eval()
        teacher = HDAV1Model(3, source).eval().requires_grad_(False)
        base = HDAV5DModel(source, teacher.adapter)
        candidate = copy.deepcopy(base)
        sx, tx = torch.randn(8, 4), torch.randn(8, 3)
        _, z = base.source_representations(sx)
        source_pools = {0:z[:4], 1:z[4:]}
        target_pools = {0:tx[:4], 1:tx[4:]}
        cfg = dict(loss_weights=V5D_WEIGHTS, adapter_lr=1e-4, classifier_lr=1e-5,
                   weight_decay=1e-4, ema_decay=.999)
        args = (teacher, [(sx, torch.tensor([0,1]*4))], [tx], source_pools,
                target_pools, torch.ones(2), cfg, {'epochs':1, 'class_batch_size':4})
        torch.manual_seed(123)
        reference = train_v5d(base, *args)
        torch.manual_seed(123)
        # Isolate the gradient trajectory from the final replacement with EMA.
        with patch('training.hda_v5n.apply_ema'):
            history, updates = train_model(candidate, *args)
        self.assertEqual(updates, 1)
        self.assertEqual(history, reference)
        for key,value in base.state_dict().items():
            torch.testing.assert_close(value, candidate.state_dict()[key], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
