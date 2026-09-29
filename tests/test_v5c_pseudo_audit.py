import unittest
import numpy as np
from evaluation.v5c_pseudo_audit import analyze_margins


class AuditTests(unittest.TestCase):
    thresholds = {"v2_margin_threshold": 0., "v5b_margin_threshold": 0.}

    def test_agreement_and_quantile_boundaries(self):
        margins = np.arange(-500., 500.)
        report, masks = analyze_margins(margins, margins, self.thresholds, 2)
        self.assertEqual(report["agreement_rate"], 1.)
        self.assertTrue(report["training_allowed"])
        self.assertEqual(int(masks["normal"].sum()), 20)
        self.assertEqual(int(masks["attack"].sum()), 30)
        self.assertFalse(masks["attack"][-1])

    def test_collapsed_normal_model_blocks_attack(self):
        margins = np.arange(-500., 500.)
        report, _ = analyze_margins(margins, np.full(1000, -1.), self.thresholds, 2)
        self.assertFalse(report["training_allowed"])
        self.assertEqual(report["pools"]["attack"]["accepted_rows"], 0)

    def test_ties_and_invalid_scores(self):
        report, _ = analyze_margins(np.zeros(100), np.zeros(100), self.thresholds, 2)
        self.assertFalse(report["training_allowed"])
        self.assertIsNone(report["cohen_kappa"])
        for a, b in (([], []), ([1], [1, 2]), ([np.nan], [0])):
            with self.assertRaises(ValueError):
                analyze_margins(a, b, self.thresholds)

    def test_normal_acceptance_uses_v5b_not_v2_threshold(self):
        margins = np.arange(1000.) + 1
        report, masks = analyze_margins(margins, -margins, self.thresholds, 2)
        self.assertEqual(int(masks["normal"].sum()), 20)
        self.assertEqual(int(masks["attack"].sum()), 0)
        self.assertFalse(report["training_allowed"])

    def test_attack_acceptance_uses_v5b_not_v2_threshold(self):
        margins = np.arange(1000.) - 1000
        report, masks = analyze_margins(margins, -margins, self.thresholds, 2)
        self.assertEqual(int(masks["attack"].sum()), 30)
        self.assertEqual(int(masks["normal"].sum()), 0)
        self.assertEqual(report["pools"]["attack"]["v5b_confirmation_rate"], 1.)


if __name__ == "__main__":
    unittest.main()
