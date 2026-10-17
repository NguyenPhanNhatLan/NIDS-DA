import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from training.proposal_class_aware import class_aware_mmd_loss
from training.proposal_mmd import load_config, train_mmd


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding=nn.Linear(2,2,bias=False)
        with torch.no_grad():
            self.embedding.weight.copy_(torch.eye(2))

    def forward(self,x):
        z=self.embedding(x)
        score=(z[:,1]-.2)*100
        return z,torch.stack([-score,score],1)


class ClassAwareMultiKernelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)
        self.source=torch.randn(8,3,requires_grad=True)
        self.target=torch.randn(8,3,requires_grad=True)
        self.labels=torch.tensor([0]*4+[1]*4)
        self.logits=torch.tensor([[8.,0.]]*4+[[0.,8.]]*4,requires_grad=True)

    def test_single_scale_matches_existing_single_kernel_loss(self):
        single,_=class_aware_mmd_loss(self.source,self.labels,self.target,self.logits,.9)
        multi,_=class_aware_mmd_loss(self.source,self.labels,self.target,self.logits,.9,[1.])
        torch.testing.assert_close(single,multi)

    def test_kernels_receive_only_same_class_samples(self):
        with patch('training.proposal_class_aware.multi_kernel_mmd_loss',
                   return_value=(torch.tensor(.2),torch.tensor(1.))) as kernel:
            loss,_=class_aware_mmd_loss(self.source,self.labels,self.target,self.logits,.9,[.5,1.,2.])
        self.assertEqual(kernel.call_count,2)
        for label,call in enumerate(kernel.call_args_list):
            torch.testing.assert_close(call.args[0],self.source[self.labels==label])
            torch.testing.assert_close(call.args[1],self.target[self.labels==label])
        self.assertAlmostEqual(loss.item(),.2,places=6)

    def test_multikernel_gradients_and_missing_class_skip(self):
        loss,_=class_aware_mmd_loss(self.source,self.labels,self.target,self.logits,.9,[.5,1.,2.])
        loss.backward()
        self.assertTrue(torch.isfinite(self.source.grad).all())
        self.assertTrue(torch.isfinite(self.target.grad).all())
        self.assertGreater(self.target.grad.norm().item(),0)
        self.assertIsNone(self.logits.grad)
        skipped,_=class_aware_mmd_loss(self.source,self.labels,self.target,torch.tensor([[8.,0.]]*8),.9,[.5,1.,2.])
        self.assertEqual(skipped.item(),0)

    def test_invalid_scales_and_kernel_metadata_fail(self):
        config=load_config('configs/proposal_class_aware_mkmmd_sep010.json')
        for scales in ([],[1.],[1.,float('nan')],[0.,1.]):
            config['mmd']['scales']=scales
            with tempfile.TemporaryDirectory() as folder:
                path=Path(folder)/'bad.json';path.write_text(json.dumps(config))
                with self.assertRaises(ValueError):
                    load_config(path)

    def test_trainer_routes_multikernel_and_records_separation(self):
        for suffix in ('','_lr0003'):
            config=load_config(f'configs/proposal_class_aware_mkmmd_sep010{suffix}.json')
            config['training']['epochs']=1
            x=torch.tensor([[1.,.1]]*4+[[1.,.3]]*4)
            model=ToyModel()
            with patch('training.proposal_mmd.evaluate_ap',side_effect=[.5,.6]), \
                 patch('training.proposal_class_aware.multi_kernel_mmd_loss',
                       wraps=__import__('training.proposal_mkmmd',fromlist=['multi_kernel_mmd_loss']).multi_kernel_mmd_loss) as kernel, \
                 redirect_stdout(io.StringIO()):
                _,_,_,history=train_mmd(model,[4,4],[(x,self.labels)],[x+.001],[],config)
            row=history[1]
            self.assertEqual(kernel.call_count,2)
            self.assertGreater(row['class_separation']['loss'],0)
            self.assertAlmostEqual(row['loss'],sum(row['loss_contributions'].values()),places=6)
            self.assertNotIn('selective_alignment',row)


if __name__ == '__main__':
    unittest.main()
