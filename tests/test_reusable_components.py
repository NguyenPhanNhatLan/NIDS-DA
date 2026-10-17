import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import torch
from torch import nn

from project_utils.statistics import seed_statistics
from training.proposal_class_aware import (
    prepare_class_batch, select_pseudo_labels, class_aware_batch_stats,
    class_aware_mmd_loss, class_separation_loss,
)
from training.proposal_mmd import load_config, train_mmd


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.embedding.weight.copy_(torch.eye(2))

    def forward(self, x):
        z = self.embedding(x)
        score = (z[:, 1] - .2) * 100
        return z, torch.stack([-score, score], 1)


class ReusableComponentTests(unittest.TestCase):
    def setUp(self):
        self.labels = torch.tensor([0, 0, 1, 1])
        self.logits = torch.tensor([[8., 0.]] * 2 + [[0., 8.]] * 2)
        self.source = torch.tensor([[1., .1]] * 2 + [[1., .3]] * 2, requires_grad=True)
        self.target = (self.source.detach() + .02).requires_grad_()

    def test_prepared_batch_reuses_selection_for_all_consumers(self):
        batch = prepare_class_batch(self.labels, self.logits, .9)
        with patch('training.proposal_class_aware.select_pseudo_labels', side_effect=AssertionError('Selection must be reused')):
            stats = class_aware_batch_stats(self.labels, self.logits, .9, batch=batch)
            mmd, _ = class_aware_mmd_loss(self.source, self.labels, self.target, self.logits, .9, batch=batch)
            separation = class_separation_loss(self.source, self.labels, self.target, self.logits, .9, .5, batch=batch)
        self.assertEqual(stats['active_classes'], [0, 1])
        (mmd + separation).backward()
        self.assertTrue(torch.isfinite(self.target.grad).all())

    def test_partial_eligibility_still_skips_both_losses(self):
        logits = torch.tensor([[8., 0.]] * 4)
        batch = prepare_class_batch(self.labels, logits, .9)
        self.assertEqual(batch.eligible_classes, [0])
        self.assertFalse(batch.can_align)
        self.assertEqual(class_aware_mmd_loss(self.source, self.labels, self.target, logits, .9, batch=batch)[0].item(), 0)
        self.assertEqual(class_separation_loss(self.source, self.labels, self.target, logits, .9, .5, batch=batch).item(), 0)

    def test_trainer_selects_pseudo_labels_once_per_batch(self):
        config = load_config('configs/proposal_class_aware_mkmmd_sep010.json')
        config['training']['epochs'] = 1
        model = ToyModel()
        with patch('training.proposal_mmd.evaluate_ap', return_value=.5), \
             patch('training.proposal_class_aware.select_pseudo_labels', wraps=select_pseudo_labels) as select, \
             redirect_stdout(io.StringIO()):
            train_mmd(model, [2, 2], [(self.source.detach(), self.labels)], [self.target.detach()], [], config)
        self.assertEqual(select.call_count, 1)

    def test_seed_statistics_alias_and_edge_cases(self):
        from evaluation.proposal_aggregate import seed_statistics as legacy_statistics
        self.assertIs(seed_statistics, legacy_statistics)
        one = seed_statistics([.3])
        self.assertEqual(one['std'], 0)
        self.assertIsNone(one['ci95'])
        for invalid in ([], [float('nan')], [[1, 2]]):
            with self.assertRaises(ValueError):
                seed_statistics(invalid)


if __name__ == '__main__':
    unittest.main()
