import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch import nn

from evaluation.calibration import (
    calibrate_margin, calibrated_probability, fit_affine_calibrator, select_threshold_by_fpr,
)
from evaluation.hda_v5b_calibration import build_frozen_target_pools, fit, report


class MarginModel(nn.Module):
    def forward(self, features):
        return features, torch.stack((torch.zeros_like(features[:, 0]), features[:, 0]), dim=1)


class CalibrationTests(unittest.TestCase):
    def test_anchor_mapping_and_positive_slope(self):
        result = fit_affine_calibrator([-3., -2., 2., 3.], [0, 0, 1, 1], [-6., -5.], [4., 5.])
        self.assertGreater(result["a"], 0)
        self.assertAlmostEqual(calibrate_margin(result["target_pseudo_normal"], result["a"], result["b"]), result["source_normal"])
        self.assertAlmostEqual(calibrate_margin(result["target_pseudo_attack"], result["a"], result["b"]), result["source_attack"])
        values = torch.arange(-10., 10.)
        probability = calibrated_probability(values, result["a"], result["b"])
        self.assertTrue(torch.all(probability[1:] > probability[:-1]))

    def test_rejects_collapsed_or_reversed_anchors(self):
        for normal, attack in (([0.], [0.]), ([2.], [1.]), ([float("nan")], [1.])):
            with self.subTest(normal=normal), self.assertRaises(ValueError):
                fit_affine_calibrator([-2., 2.], [0, 1], normal, attack)

    def test_fpr_threshold_matches_brute_force_with_ties(self):
        rng = np.random.default_rng(42)
        for max_fpr in (0.0, 0.01, 0.2, 1.0):
            for _ in range(10):
                scores = rng.integers(-5, 6, 50).astype(float)
                labels = np.array([0, 1] * 25)
                candidates = np.r_[np.nextafter(scores.max(), np.inf), np.unique(scores)[::-1]]
                feasible = []
                for threshold in candidates:
                    predictions = scores >= threshold
                    recall = predictions[labels == 1].mean()
                    fpr = predictions[labels == 0].mean()
                    if fpr <= max_fpr:
                        feasible.append((recall, -fpr, threshold))
                expected = max(feasible)[2]
                self.assertEqual(select_threshold_by_fpr(labels, scores, max_fpr), expected)
        threshold = select_threshold_by_fpr([0, 1], [1., 1.], max_fpr=0)
        self.assertGreater(threshold, 1.)

    def test_fit_without_development_then_report_frozen_values(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path, target_path, dev_path = (root / name for name in ("source", "train", "dev"))
            source_path.mkdir()
            target_path.mkdir()
            source_values = np.linspace(-3., 3., 128)
            pq.write_table(pa.table({"features": [[float(x)] for x in source_values],
                                     "label": (source_values > 0).astype(int).tolist()}), source_path / "rows.parquet")
            pq.write_table(pa.table({"features": [[float(x)] for x in np.linspace(-4., 4., 256)]}), target_path / "rows.parquet")
            reference = {
                "source_validation": str(source_path), "checkpoint_sha256": "fixed-checkpoint",
                "threshold_policy": {"max_fpr": 0.01},
            }
            reference_path = root / "reference.json"
            reference_path.write_text(json.dumps(reference))
            protocol = {"training": {"batch_size": 32},
                        "pseudo_labels": {"normal_quantile": 0.02, "attack_quantile_low": 0.95,
                                          "attack_quantile_high_exclusive": 0.98, "dynamic_updates": False},
                        "target_data": {"adaptation_train": str(target_path), "development": str(dev_path)}}
            provenance = {"source_dim": 1, "target_dim": 1}
            models = (reference, protocol, provenance, MarginModel(), MarginModel(), MarginModel())
            calibration_path, output_path = root / "calibration.json", root / "report.json"
            with patch("evaluation.hda_v5b_calibration.load_frozen_models", return_value=models), \
                    contextlib.redirect_stdout(io.StringIO()):
                fit(reference_path, calibration_path)
                self.assertFalse(dev_path.exists())
                frozen_bytes = calibration_path.read_bytes()
                frozen = json.loads(frozen_bytes)["frozen"]
                self.assertFalse(frozen["target_labels_used_for_fit"])
                self.assertEqual(frozen["pseudo_label_metadata"]["train_rows"], 256)
                self.assertFalse(frozen["pseudo_label_metadata"]["labels_used"])
                self.assertLessEqual(frozen["source_validation_metrics"]["fpr"], 0.01)
                with self.assertRaises(FileExistsError):
                    fit(reference_path, calibration_path)
                dev_path.mkdir()
                pq.write_table(pa.table({"features": [[-2.], [-1.], [1.], [2.]], "label": [0, 0, 1, 1]}), dev_path / "rows.parquet")
                report(calibration_path, output_path)
                self.assertEqual(calibration_path.read_bytes(), frozen_bytes)
                metrics = json.loads(output_path.read_text())
                self.assertEqual(metrics["roc_auc"], metrics["raw_margin_roc_auc"])
                self.assertEqual(metrics["pr_auc"], metrics["raw_margin_pr_auc"])
                self.assertNotIn("oracle", metrics)
                modified = json.loads(frozen_bytes)
                modified["frozen"]["b"] += 1
                calibration_path.write_text(json.dumps(modified))
                with self.assertRaisesRegex(ValueError, "modified"):
                    report(calibration_path, root / "other_report.json")

    def test_rejects_changed_pseudo_policy(self):
        protocol = {"pseudo_labels": {"normal_quantile": 0.05, "attack_quantile_low": 0.95,
                                      "attack_quantile_high_exclusive": 0.98, "dynamic_updates": False}}
        with self.assertRaisesRegex(ValueError, "frozen"):
            build_frozen_target_pools(MarginModel(), protocol, 1, 32)


if __name__ == "__main__":
    unittest.main()
