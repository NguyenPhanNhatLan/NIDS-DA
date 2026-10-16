import copy
import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import torch

from models.baseline import BaselineMLP
from training.proposal_class_aware import class_separation_loss
from training.proposal_mmd import load_config, train_mmd


class ClassSeparationTests(unittest.TestCase):
    def setUp(self):
        self.labels = torch.tensor([0, 0, 1, 1])
        self.logits = torch.tensor([[8., 0.], [8., 0.], [0., 8.], [0., 8.]], requires_grad=True)

    def loss(self, source, target, logits=None):
        return class_separation_loss(source, self.labels, target,
                                     self.logits if logits is None else logits, .9, .5)

    def test_well_separated_classes_have_zero_penalty(self):
        x = torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., 1.]])
        self.assertEqual(self.loss(x, x).item(), 0)

    def test_opposite_class_cross_domain_overlap_is_penalized(self):
        x = torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., 1.]])
        swapped = x.flip(0)
        self.assertGreater(self.loss(x, swapped).item(), 0)

    def test_gradient_step_reduces_overlap_penalty(self):
        x = torch.tensor([[1., .1], [1., .1], [1., .3], [1., .3]], requires_grad=True)
        y = x.detach().clone().requires_grad_()
        before = self.loss(x, y)
        before.backward()
        self.assertIsNone(self.logits.grad)
        self.assertTrue(torch.isfinite(x.grad).all())
        after = self.loss(x - .1*x.grad, y - .1*y.grad)
        self.assertLess(after.item(), before.item())

    def test_missing_or_unconfident_class_skips_batch(self):
        x = torch.randn(4, 3, requires_grad=True)
        for logits in (torch.zeros(4, 2), torch.tensor([[8., 0.]]*4)):
            loss = self.loss(x, x, logits)
            self.assertEqual(loss.item(), 0)
            loss.backward()
            self.assertTrue(torch.isfinite(x.grad).all())

    def test_trainer_logs_and_weights_separation(self):
        config = copy.deepcopy(load_config('configs/proposal_class_aware_dev_confidence90_lambda001_separation.json'))
        config['training']['epochs'] = 1
        config['mmd']['target_pseudo_label_confidence'] = 0.5
        model = BaselineMLP(5)
        x = torch.randn(8, 5)
        y = torch.tensor([0, 1]*4)
        with patch('training.proposal_mmd.evaluate_ap', return_value=.5), redirect_stdout(io.StringIO()):
            _, _, _, history = train_mmd(model, [4, 4], [(x,y)], [x.clone()], [], config)
        row = history[1]
        self.assertIn('class_separation', row)
        expected = row['source_ce'] + .001*row['mmd2'] + .001*row['class_separation']['loss']
        self.assertAlmostEqual(row['loss'], expected, places=6)


if __name__ == '__main__':
    unittest.main()
