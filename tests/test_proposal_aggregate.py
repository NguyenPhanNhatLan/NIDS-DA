import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluation import proposal_aggregate


class ProposalAggregateTests(unittest.TestCase):
    def test_aggregates_seed_metrics_and_gain(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            direction = "unsw_to_cicids"
            for seed in (42, 43):
                for method in ("source_only", *proposal_aggregate.CONFIGS):
                    ap = 0.2 + (seed - 42) * 0.1 + (0 if method == "source_only" else 0.05)
                    metrics = {"pr_auc": ap, "macro_f1": 0.4, "recall": 0.5,
                               "fpr": 0.1, "roc_auc": 0.6,
                               "tn": 8, "fp": 1, "fn": 1, "tp": 2}
                    result = {"direction": direction, "seed": seed,
                              "target_development_split": "cicids_val",
                              "common_feature_config_sha256": "config",
                              "features": list(proposal_aggregate.COMMON_FEATURES), "feature_count": 5,
                              "preprocessor_sha256": "processor",
                              "prepared_split_sha256": {"train": "data"}}
                    if method == "source_only":
                        result.update(target_development=metrics, checkpoint=f"source-{seed}")
                    else:
                        result.update(cross_domain=metrics, source_checkpoint=f"source-{seed}",
                                      selected_stage="adaptation", config_sha256="config")
                    (root / f"{method}-{seed}.json").write_text(json.dumps(result))
            def fake_path(method, direction, seed):
                return root / f"{method}-{seed}.json"
            with patch.object(proposal_aggregate, "result_path", side_effect=fake_path), \
                 patch.object(proposal_aggregate, "diagnostic_path", return_value=root / "missing.json"), \
                 patch.object(proposal_aggregate, "sha256", return_value="config"):
                output = proposal_aggregate.aggregate((direction,), (42, 43))
            summary = next(row for row in output["summary"] if row["method"] == "marginal_mmd")
            self.assertAlmostEqual(summary["metrics"]["pr_auc"]["mean"], 0.30)
            self.assertAlmostEqual(summary["metrics"]["adaptation_gain_pr_auc"]["mean"], 0.05)
            self.assertEqual([row["seed"] for row in summary["paired_delta_ap"]], [42, 43])
            self.assertAlmostEqual(summary["metrics"]["adaptation_gain_pr_auc"]["ci95"][0], 0.05)
            self.assertEqual(summary["metrics"]["pr_auc"]["n"], 2)
            self.assertEqual(summary["confusion_matrix_sum"], {"tn": 16, "fp": 2, "fn": 2, "tp": 4})

    def test_ci_uses_paired_seed_differences_and_sample_std(self):
        # Five paired deltas: mean .02, sample std sqrt(.00025).
        result = proposal_aggregate.seed_statistics([0, .01, .02, .03, .04])
        self.assertAlmostEqual(result["mean"], .02)
        self.assertAlmostEqual(result["std"], .015811388300841896)
        self.assertAlmostEqual(result["ci95"][0], .000367568385224389, places=8)
        self.assertAlmostEqual(result["ci95"][1], .03963243161477561, places=8)
        self.assertIsNone(proposal_aggregate.seed_statistics([.02])["ci95"])
        with self.assertRaises(ValueError):
            proposal_aggregate.aggregate(seeds=(42, 42))


if __name__ == "__main__":
    unittest.main()
