import unittest

import torch

from models.baseline import BaselineMLP
from models.hda_v1 import HDAV1Model
from training.hda_v5c import train_adapter


class V5cTrainingTests(unittest.TestCase):
    def test_only_student_adapter_changes_and_balanced_passes_do_not_update_bn(self):
        torch.manual_seed(42)
        source = BaselineMLP(4).eval()
        teacher = HDAV1Model(3, source).eval()
        source_before = {k: v.clone() for k, v in source.state_dict().items()}
        teacher_before = {k: v.clone() for k, v in teacher.state_dict().items()}
        source_x, target_x = torch.randn(8, 4), torch.randn(8, 3)
        with torch.no_grad():
            latent = torch.relu(source.bn2(source.fc2(source.encode_hidden(source_x))))
        student, history = train_adapter(
            source, teacher, [(source_x, torch.zeros(8, dtype=torch.long))], [target_x],
            {0: latent[:4], 1: latent[4:]}, {0: target_x[:4], 1: target_x[4:]},
            {"normal": target_x[:4], "attack": target_x[4:]},
            {"learning_rate": 0.001, "epochs": 1, "class_batch_size": 4}, 0.10)
        for k, v in source.state_dict().items():
            self.assertTrue(torch.equal(v, source_before[k]), k)
        for k, v in teacher.state_dict().items():
            self.assertTrue(torch.equal(v, teacher_before[k]), k)
        self.assertFalse(torch.equal(student.adapter.layers[0].weight, teacher.adapter.layers[0].weight))
        self.assertEqual(student.adapter.layers[1].num_batches_tracked.item(),
                         teacher.adapter.layers[1].num_batches_tracked.item() + 1)
        self.assertGreater(history[0]["pl"], 0)
        self.assertTrue(all(not p.requires_grad for p in teacher.parameters()))
        self.assertTrue(all(not p.requires_grad for p in source.parameters()))


if __name__ == "__main__":
    unittest.main()
