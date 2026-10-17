import copy
import io
import json
import tempfile
from pathlib import Path
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import torch
from torch import nn

from training.proposal_class_aware import (
    confidence_setting, select_pseudo_labels, class_aware_batch_stats,
    class_separation_loss, class_aware_mmd_loss,
)
from training.proposal_mmd import load_config, train_mmd, weighted_gradient_norm
from evaluation.proposal_alignment_audit import pseudo_quality


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Linear(2,2,bias=False)
        with torch.no_grad():
            self.embedding.weight.copy_(torch.eye(2))

    def forward(self,x):
        z=self.embedding(x)
        score=100*(z[:,1]-.2)
        return z,torch.stack([-score,score],1)


class AblationTests(unittest.TestCase):
    def setUp(self):
        self.x=torch.tensor([[1.,.1],[1.,.1],[1.,.3],[1.,.3]])
        self.y=torch.tensor([0,0,1,1])

    def test_removed_selective_config_cannot_silently_run(self):
        config=load_config('configs/proposal_class_aware_ablation_sep010.json')
        config['selective_alignment']={'teacher_decay':.99}
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'retired.json'
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError,'has been removed'):
                load_config(path)

    def test_scalar_selection_is_backward_compatible(self):
        logits=torch.randn(30,2)
        for threshold in (.8,.9):
            labels,accepted=select_pseudo_labels(logits,threshold)
            probabilities,expected=logits.softmax(1).max(1)
            self.assertTrue(torch.equal(labels,expected))
            self.assertTrue(torch.equal(accepted,probabilities>=threshold))
            _,mapped=select_pseudo_labels(logits,{'0':threshold,'1':threshold})
            self.assertTrue(torch.equal(mapped,accepted))

    def test_asymmetric_threshold_changes_eligibility_in_both_losses(self):
        logits=torch.tensor([[.92,.08],[.92,.08],[.08,.92],[.08,.92]]).log()
        thresholds={'0':.9,'1':.95}
        stats=class_aware_batch_stats(self.y,logits,thresholds)
        self.assertEqual(stats['accepted_per_class'],[2,0])
        self.assertEqual(stats['active_classes'],[])
        s=self.x.clone().requires_grad_()
        self.assertEqual(class_separation_loss(s,self.y,s,logits,thresholds,.5).item(),0)
        self.assertEqual(class_aware_mmd_loss(s,self.y,s,logits,thresholds)[0].item(),0)
        quality=pseudo_quality(logits.exp().numpy(),self.y.numpy(),thresholds)
        self.assertEqual(quality['accepted_attack'],0)

    def test_invalid_class_thresholds_fail(self):
        for thresholds in ({'0':.9},{'0':.9,'1':float('nan')},{'0':.9,'1':1.1}):
            with self.assertRaises(ValueError):
                confidence_setting({'mmd':{'target_pseudo_label_confidence_by_class':thresholds}})

    def test_weighted_gradient_norm_is_correct_and_does_not_accumulate(self):
        p=nn.Parameter(torch.tensor([3.,4.]))
        term=.5*p.square().sum()
        self.assertAlmostEqual(weighted_gradient_norm(term,(p,)),5.)
        self.assertIsNone(p.grad)
        term.backward()
        torch.testing.assert_close(p.grad,torch.tensor([3.,4.]))

    def train_once(self,gradient_every):
        config=copy.deepcopy(load_config('configs/proposal_class_aware_ablation_sep_only.json'))
        config['training']['epochs']=1
        config['diagnostics']['gradient_every_steps']=gradient_every
        model=TinyModel()
        with patch('training.proposal_mmd.evaluate_ap',side_effect=[.5,.6]), \
             patch('training.proposal_mmd.class_aware_mmd_loss',side_effect=AssertionError('MMD must be disabled')), \
             redirect_stdout(io.StringIO()):
            _,_,_,history=train_mmd(model,[2,2],[(self.x,self.y)],[self.x],[],config)
        return model,history[1]

    def test_no_mmd_keeps_separation_and_gradient_diagnostics(self):
        _,row=self.train_once(1)
        self.assertEqual(row['mmd2'],0)
        self.assertEqual(row['class_aware_alignment']['aligned_batches'],0)
        self.assertEqual(row['class_separation']['eligible_batches'],1)
        self.assertGreater(row['class_separation']['loss'],0)
        norms=row['gradient_diagnostics']['mean_weighted_parameter_gradient_l2']
        self.assertEqual(norms['class_mmd'],0)
        self.assertGreater(norms['class_separation'],0)
        self.assertAlmostEqual(row['loss'],sum(row['loss_contributions'].values()),places=6)

    def test_diagnostics_leave_parameter_update_unchanged(self):
        before,_=self.train_once(0)
        after,_=self.train_once(1)
        for a,b in zip(before.parameters(),after.parameters()):
            torch.testing.assert_close(a,b,rtol=0,atol=0)


if __name__ == '__main__':
    unittest.main()
