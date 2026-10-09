import unittest
from unittest.mock import patch
import tempfile
from pathlib import Path

import numpy as np
import torch
import pyarrow as pa
import pyarrow.parquet as pq

from models.baseline import BaselineMLP
from training import proposal_mmd
from training.proposal_class_aware import class_aware_mmd_loss, pseudo_label_counts
from training.proposal_mkmmd import multi_kernel_mmd_loss
from training.proposal_data import ParquetBatchStream


class ProposalAlignmentTests(unittest.TestCase):
    def test_uda_training_reads_target_features_without_label_column(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            values = np.random.default_rng(2).normal(size=(8, 5)).astype(np.float32)
            vectors = pa.FixedSizeListArray.from_arrays(pa.array(values.ravel()), 5)
            source = root / "source"
            target = root / "target"
            source.mkdir(); target.mkdir()
            pq.write_table(pa.table({"features": vectors,
                                     "label": pa.array([0, 1] * 4)}), source / "data.parquet")
            pq.write_table(pa.table({"features": vectors}), target / "data.parquet")
            source_stream = ParquetBatchStream(source, 4, True, 42, True, drop_last=True)
            target_stream = ParquetBatchStream(target, 4, True, 43, False, drop_last=True)
            config = {"method": "marginal_mmd", "alpha_ce": 1.0, "lambda_mmd": 0.0,
                      "training": {"epochs": 1, "learning_rate": 0.001,
                                   "weight_decay": 0.0, "min_delta": 0.0001, "patience": 2}}
            with patch.object(proposal_mmd, "evaluate_ap", side_effect=[0.5, 0.6]):
                _, epoch, _, _ = proposal_mmd.train_mmd(
                    BaselineMLP(5), [4, 4], source_stream, target_stream, [], config)
            self.assertEqual(epoch, 1)

    def test_multikernel_alignment_has_encoder_gradient(self):
        source = torch.randn(8, 5, requires_grad=True)
        target = torch.randn(8, 5) + 1
        loss, bandwidth = multi_kernel_mmd_loss(source, target, [0.5, 1.0, 2.0])
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(bandwidth), 0)
        loss.backward()
        self.assertGreater(float(source.grad.norm()), 0)

    def test_class_aware_uses_pseudo_labels_only(self):
        source = torch.randn(8, 5, requires_grad=True)
        target = torch.randn(8, 5, requires_grad=True)
        source_labels = torch.tensor([0] * 4 + [1] * 4)
        target_logits = torch.tensor([[5.0, 0.0]] * 4 + [[0.0, 5.0]] * 4)
        loss, bandwidth = class_aware_mmd_loss(source, source_labels, target, target_logits, 0.8)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(bandwidth), 0)
        loss.backward()
        self.assertGreater(float(source.grad.norm()), 0)
        self.assertGreater(float(target.grad.norm()), 0)
        zero, _ = class_aware_mmd_loss(source, source_labels, target,
                                       torch.zeros_like(target_logits), 0.9)
        self.assertEqual(float(zero.detach()), 0.0)

    def test_class_aware_skips_normal_only_batch(self):
        source = torch.randn(4, 5, requires_grad=True)
        target = torch.randn(4, 5, requires_grad=True)
        source_y = torch.tensor([0, 0, 1, 1])
        logits = torch.tensor([[10., -10.]] * 4)
        loss, bandwidth = class_aware_mmd_loss(source, source_y, target, logits, .8)
        self.assertEqual(float(loss), 0.)
        self.assertEqual(float(bandwidth), 0.)
        loss.backward()
        self.assertIsNotNone(source.grad)

    def test_pseudo_label_acceptance_counts_only_confident_predictions(self):
        logits = torch.tensor([[5.0, 0.0], [0.0, 5.0], [0.0, 0.0]])
        self.assertEqual(pseudo_label_counts(logits, 0.8), (2, 3, [1, 1]))

    def test_train_dispatches_mk_and_class_aware(self):
        x = torch.randn(8, 5)
        y = torch.tensor([0, 1] * 4)
        for method, function_name, mmd_config in (
            ("mk_mmd", "multi_kernel_mmd_loss", {"scales": [0.5, 1.0, 2.0]}),
            ("class_aware_mmd", "class_aware_mmd_loss", {"target_pseudo_label_confidence": 0.8}),
        ):
            model = BaselineMLP(5)
            config = {"method": method, "alpha_ce": 1.0, "lambda_mmd": 0.01,
                      "mmd": mmd_config, "training": {"epochs": 1, "learning_rate": 0.001,
                      "weight_decay": 0.0, "min_delta": 0.0001, "patience": 2}}
            def fake_alignment(*args):
                return args[0].mean().square(), args[0].new_tensor(1.0)
            with patch.object(proposal_mmd, "evaluate_ap", side_effect=[0.5, 0.6]), \
                 patch.object(proposal_mmd, function_name, side_effect=fake_alignment) as alignment:
                _, best_epoch, _, history = proposal_mmd.train_mmd(
                    model, [4, 4], [(x, y)], [x], [(x, y)], config)
            alignment.assert_called_once()
            self.assertEqual(best_epoch, 1)
            self.assertAlmostEqual(history[1]["loss"], history[1]["source_ce"]
                                   + 0.01 * history[1]["mmd2"], places=6)
            if method == "class_aware_mmd":
                acceptance = history[1]["pseudo_label_acceptance"]
                self.assertEqual(acceptance["seen"], len(x))
                self.assertEqual(sum(acceptance["accepted_per_class"]), acceptance["accepted"])


if __name__ == "__main__":
    unittest.main()
