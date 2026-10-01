"""Synthetic V5l invariants; created for the user to run, not executed on authoring."""
import unittest
from unittest.mock import patch

import torch

from models.baseline import BaselineMLP
from models.hda_v1 import HDAV1Model
from training.hda_v5l import V5B_WEIGHTS, train_v5l


class V5lTests(unittest.TestCase):
    def test_v5b_initialization_and_only_adapter_changes(self):
        torch.manual_seed(42)
        source = BaselineMLP(4).eval()
        v5b = HDAV1Model(3, source).eval().requires_grad_(False)
        with torch.no_grad():
            v5b.adapter.layers[0].weight.add_(.2)
        classifier_before = {k: v.clone() for k, v in source.classifier.state_dict().items()}
        teacher_before = {k: v.clone() for k, v in v5b.state_dict().items()}
        x, t = torch.randn(8, 4), torch.randn(8, 3)
        with torch.no_grad():
            z, _ = source(x)
        original_load = torch.nn.Module.load_state_dict
        loaded = []

        def capture(module, state, *args, **kwargs):
            result = original_load(module, state, *args, **kwargs)
            if isinstance(module, type(v5b.adapter)):
                loaded.append({k: v.clone() for k, v in module.state_dict().items()})
            return result

        with patch.object(torch.nn.Module, "load_state_dict", capture):
            student, history = train_v5l(
                source, v5b, [(x, torch.tensor([0, 1] * 4))], [t],
                {0: z[:4], 1: z[4:]}, {0: t[:4], 1: t[4:]},
                V5B_WEIGHTS, epochs=1, class_batch_size=4)
        self.assertEqual(len(loaded), 1)
        for key, value in v5b.adapter.state_dict().items():
            torch.testing.assert_close(loaded[0][key], value, rtol=0, atol=0)
        for key, value in student.classifier.state_dict().items():
            torch.testing.assert_close(value, classifier_before[key], rtol=0, atol=0)
        for key, value in v5b.state_dict().items():
            torch.testing.assert_close(value, teacher_before[key], rtol=0, atol=0)
        self.assertTrue(any(not torch.equal(p, q) for p, q in zip(student.adapter.parameters(), v5b.adapter.parameters())))
        self.assertTrue(all(p.grad is None for p in student.classifier.parameters()))
        self.assertEqual(student.adapter.layers[1].num_batches_tracked.item(), 1)
        row = history[0]
        self.assertAlmostEqual(row["loss"], sum(V5B_WEIGHTS[k] * row[k] for k in V5B_WEIGHTS), places=5)


if __name__ == "__main__":
    unittest.main()
