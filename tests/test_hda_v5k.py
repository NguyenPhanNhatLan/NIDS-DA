import unittest
from unittest.mock import patch
import torch

from models.baseline import BaselineMLP
from models.hda_v1 import HDAV1Model
from training.hda_v5k import train_hda_v5k, V5B_WEIGHTS
from training.latent_vat_v2 import latent_vat_loss
from training.adaptation import mmd_loss


class V5kTests(unittest.TestCase):
    def test_one_step_updates_adapter_only_and_preserves_bn_policy(self):
        torch.manual_seed(42)
        source = BaselineMLP(4).eval()
        teacher = HDAV1Model(3, source).eval().requires_grad_(False)
        x, t = torch.randn(8, 4), torch.randn(8, 3)
        with torch.no_grad():
            z, _ = source(x)
        old_source = {k:v.clone() for k,v in source.state_dict().items()}
        old_teacher = {k:v.clone() for k,v in teacher.state_dict().items()}
        vat = dict(weight=.1, epsilon_ratio=.05, xi_ratio=.001, power_iterations=1)
        with patch('training.hda_v5k.latent_vat_loss', wraps=latent_vat_loss) as vat_call, \
             patch('training.hda_v5k.mmd_loss', wraps=mmd_loss) as mmd_call:
            model, history = train_hda_v5k(source, teacher,
                [(x,torch.tensor([0,1]*4))], [t], {0:z[:4],1:z[4:]},
                {0:t[:4],1:t[4:]}, V5B_WEIGHTS, vat, epochs=1, class_batch_size=3)
        self.assertEqual(vat_call.call_count, 1)
        self.assertEqual(mmd_call.call_count, 3)
        vat_z = vat_call.call_args.args[0]
        self.assertEqual(len(vat_z), 6)  # Balanced anchors, not the natural batch of 8.
        torch.testing.assert_close(vat_z[:3], mmd_call.call_args_list[1].args[1])
        torch.testing.assert_close(vat_z[3:], mmd_call.call_args_list[2].args[1])
        row = history[0]
        self.assertAlmostEqual(row['loss'], sum(V5B_WEIGHTS[k]*row[k] for k in V5B_WEIGHTS)+vat['weight']*row['vat'], places=5)
        for k,v in source.state_dict().items(): torch.testing.assert_close(v,old_source[k],rtol=0,atol=0)
        for k,v in teacher.state_dict().items(): torch.testing.assert_close(v,old_teacher[k],rtol=0,atol=0)
        self.assertTrue(all(p.grad is None for p in source.parameters()))
        self.assertTrue(any(not torch.equal(p,q) for p,q in zip(model.adapter.parameters(),teacher.adapter.parameters())))
        self.assertEqual(model.adapter.layers[1].num_batches_tracked.item(),1)


if __name__ == '__main__': unittest.main()
