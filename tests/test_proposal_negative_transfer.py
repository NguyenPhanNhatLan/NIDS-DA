import unittest

import numpy as np

from evaluation.proposal_negative_transfer import (
    class_conditional_mmd,
    diagnose,
    label_prior_shift,
    repeated_mmd,
    threshold_diagnostic,
)
from evaluation.baseline import compute_metrics


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
        active = diagnose(delta, 0, marginal, classes, gradient, domain, prior, threshold, True)
        inactive = diagnose(delta, 0, marginal, classes, gradient, domain, prior, threshold, False)
        self.assertIn("ce_mmd_gradient_conflict", active)
        self.assertNotIn("ce_mmd_gradient_conflict", inactive)
        self.assertNotIn("weak_alignment", inactive)


if __name__ == "__main__":
    unittest.main()
