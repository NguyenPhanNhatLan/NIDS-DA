import unittest

import numpy as np
import torch
from unittest.mock import patch

from evaluation.proposal_negative_transfer import (
    class_conditional_mmd,
    diagnose,
    gradient_cosine,
    label_prior_shift,
    repeated_mmd,
    threshold_diagnostic,
)
from evaluation.baseline import compute_metrics
from evaluation import proposal_negative_transfer as diagnostic_module
from models.baseline import BaselineMLP


class NegativeTransferDiagnosticTests(unittest.TestCase):
    def test_mmd_prior_and_threshold_are_finite(self):
        rng = np.random.default_rng(42)
        source = rng.normal(size=(20, 4)).astype(np.float32)
        target = (source + 0.2).astype(np.float32)
        labels = np.array([0] * 10 + [1] * 10)
        mmd = repeated_mmd(source, target, sample_size=8, repeats=3, seed=7)
        self.assertEqual(mmd["repeats"], 3)
        self.assertTrue(np.isfinite(mmd["mmd2_mean"]))
        classes = class_conditional_mmd(source, labels, target, labels, sample_size=5, seed=7)
        self.assertEqual(classes["sample_size_per_class"], 5)
        self.assertGreater(classes["shared_bandwidth"], 0)
        self.assertTrue(np.isfinite(classes["separation_ratio"]))
        prior = label_prior_shift(labels, np.array([0, 1, 1, 1]))
        self.assertEqual(prior["absolute_gap"], 0.25)
        scores = np.array([0.1, 0.4, 0.6, 0.9])
        y = np.array([0, 1, 0, 1])
        metrics = compute_metrics(y, scores, 0.5)
        diagnostic = threshold_diagnostic(y, scores, 0.5, metrics)
        self.assertIn("target_oracle_f1_development_only", diagnostic)

    def test_lambda_zero_does_not_claim_mmd_conflict(self):
        delta = {"pr_auc": -0.2}
        marginal = {"relative_mmd_reduction": 0.5}
        classes = {"before": {"separation_ratio": 2.0}, "after": {"separation_ratio": 1.0}}
        gradient = {"mean": -0.5, "negative_fraction": 0.9}
        domain = {"after": {"auc_mean": 0.9}}
        prior = {"absolute_gap": 0.0}
        threshold = {"after": {"threshold_issue": False}}
        bn = {"after": {"bn1": {"source_shift": 0.2, "target_shift": 2.0, "target_minus_source": 1.8}}}
        active = diagnose(delta, 0.1, marginal, classes, gradient, domain, prior, threshold, True, bn)
        inactive = diagnose(delta, 0, marginal, classes, gradient, domain, prior, threshold, False)
        self.assertIn("ce_mmd_gradient_conflict", active)
        self.assertIn("mmd_reduced_but_domains_still_separable", active)
        self.assertNotIn("possible_over_alignment", active)
        self.assertIn("source_over_specialization", active)
        self.assertIn("normalization_shift", active)
        self.assertNotIn("ce_mmd_gradient_conflict", inactive)
        self.assertNotIn("weak_alignment", inactive)
        aligned_domain = {"after": {"auc_mean": 0.55}}
        aligned = diagnose(delta, 0, marginal, classes, gradient,
                           aligned_domain, prior, threshold, True)
        self.assertIn("possible_over_alignment", aligned)
        self.assertNotIn("weak_alignment", aligned)

    def test_gradient_uses_dropout_training_with_bn_frozen(self):
        model = BaselineMLP(10)
        model.eval()
        x = torch.randn(8, 10)
        y = torch.tensor([0, 1] * 4)
        original_mmd = diagnostic_module.mmd_loss

        def checked_mmd(source, target):
            self.assertTrue(model.training)
            self.assertTrue(model.dropout.training)
            self.assertFalse(model.bn1.training)
            self.assertFalse(model.bn2.training)
            return original_mmd(source, target)

        def fake_stream(path, batch_size, shuffle, seed, include_labels, drop_last=False):
            return [(x, y)] if include_labels else [x + 0.5]

        with patch.object(diagnostic_module, "ParquetBatchStream", side_effect=fake_stream), \
             patch.object(diagnostic_module, "mmd_loss", side_effect=checked_mmd):
            result = gradient_cosine(model, "source", "target", 10, [4, 4], batches=1, batch_size=8)
        self.assertEqual(result["batches"], 1)
        self.assertFalse(model.training)
        self.assertFalse(model.dropout.training)
        self.assertFalse(model.bn1.training)

    def test_class_mmd_uses_one_shared_bandwidth(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(20, 4)).astype(np.float32)
        y = np.array([0] * 10 + [1] * 10)
        used = []
        original_kernel = diagnostic_module.rbf_kernel

        def record_kernel(a, b, bandwidth):
            used.append(float(bandwidth))
            return original_kernel(a, b, bandwidth)

        with patch.object(diagnostic_module, "estimate_bandwidth_squared", return_value=torch.tensor(2.0)) as estimate, \
             patch.object(diagnostic_module, "rbf_kernel", side_effect=record_kernel):
            class_conditional_mmd(x, y, x + 0.1, y, sample_size=5)
        estimate.assert_called_once()
        self.assertEqual(len(used), 12)
        self.assertEqual(set(used), {2.0})


if __name__ == "__main__":
    unittest.main()
