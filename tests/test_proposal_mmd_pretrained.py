import unittest
from unittest.mock import patch
import json
import tempfile
from pathlib import Path

import torch

from models.baseline import BaselineMLP
from training import proposal_mmd


class ProposalMmdPretrainedTests(unittest.TestCase):
    def test_config_rejects_target_test_development(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "mmd.json"
            path.write_text(json.dumps({"feature_count": 10, "lambda_mmd": 0.01, "alpha_ce": 1.0,
                                        "development_split": "target_test"}))
            with self.assertRaisesRegex(ValueError, "development_split=target_val"):
                proposal_mmd.load_config(path)

    def test_stale_preprocessor_hash_fails_before_training(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            direction = "unsw_to_cicids"
            preprocessor = root / "models/proposal_v1" / direction / "preprocessor.joblib"
            preprocessor.parent.mkdir(parents=True)
            preprocessor.write_bytes(b"current-preprocessor")
            source_cp_path = root / "source_models" / direction / "seed42.pt"
            source_result_path = root / "source_results" / direction / "seed42.json"
            source_cp_path.parent.mkdir(parents=True)
            source_result_path.parent.mkdir(parents=True)
            source_result_path.write_text(json.dumps({
                "direction": direction, "seed": 42, "feature_count": 10,
                "features": list(proposal_mmd.COMMON_FEATURES),
                "best_epoch": 1,
                "target_development_split": "cicids_val",
            }))
            torch.save({
                "direction": direction, "seed": 42, "input_dim": 10,
                "features": list(proposal_mmd.COMMON_FEATURES), "best_epoch": 1,
                "common_feature_config_sha256": "stale",
                "preprocessor_sha256": "stale",
            }, source_cp_path)
            with patch.object(proposal_mmd, "ROOT", root), \
                 patch.object(proposal_mmd, "SOURCE_ONLY_ROOT", root / "source_results"), \
                 patch.object(proposal_mmd, "SOURCE_CHECKPOINT_ROOT", root / "source_models"), \
                 patch.object(proposal_mmd, "output_paths", return_value=(root / "new.pt", root / "new.json")):
                with self.assertRaisesRegex(ValueError, "checkpoint does not match"):
                    proposal_mmd.run(direction, 42, proposal_mmd.DEFAULT_CONFIG)

    def test_epoch_zero_ap_must_match_source_checkpoint(self):
        model = BaselineMLP(10)
        config = {"lambda_mmd": 0.0, "alpha_ce": 1.0, "training": {
            "epochs": 1, "learning_rate": 0.001, "weight_decay": 0.0,
            "min_delta": 0.0001, "patience": 2,
        }}
        with patch.object(proposal_mmd, "evaluate_ap", return_value=0.4):
            with self.assertRaisesRegex(ValueError, "Source checkpoint no longer matches"):
                proposal_mmd.train_mmd(model, [2, 2], [], [], [], config,
                                       expected_source_ap=0.8)

    def test_lambda_zero_is_ce_only_and_bn_stats_stay_frozen(self):
        model = BaselineMLP(10)
        initial_state = {key: value.clone() for key, value in model.state_dict().items()}
        before = [(m.running_mean.clone(), m.running_var.clone())
                  for m in model.modules() if isinstance(m, torch.nn.BatchNorm1d)]
        x = torch.randn(4, 10)
        y = torch.tensor([0, 1, 0, 1])
        config = {"lambda_mmd": 0.0, "alpha_ce": 1.0, "training": {
            "epochs": 1, "learning_rate": 0.001, "weight_decay": 0.0,
            "min_delta": 0.0001, "patience": 2,
        }}
        with patch.object(proposal_mmd, "evaluate_ap", side_effect=[0.6, 0.5]), \
             patch.object(proposal_mmd, "mmd_loss", side_effect=AssertionError("MMD must not run")):
            trained, epoch, ap, history = proposal_mmd.train_mmd(
                model, [2, 2], [(x, y)], [x], [(x, y)], config
            )
        self.assertIs(trained, model)
        self.assertEqual(epoch, 0)
        self.assertEqual(ap, 0.6)
        self.assertEqual(history[0]["stage"], "source_pretrained")
        self.assertEqual(history[0]["source_val_ap"], 0.6)
        self.assertIsNone(history[0]["mmd2"])
        self.assertEqual(history[1]["mmd2"], 0.0)
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, initial_state[key]), key)
        after = [(m.running_mean, m.running_var)
                 for m in model.modules() if isinstance(m, torch.nn.BatchNorm1d)]
        for (mean_before, var_before), (mean_after, var_after) in zip(before, after):
            self.assertTrue(torch.equal(mean_before, mean_after))
            self.assertTrue(torch.equal(var_before, var_after))

    def test_positive_lambda_calls_mmd_and_uses_separate_paths(self):
        model = BaselineMLP(10)
        x = torch.randn(4, 10)
        y = torch.tensor([0, 1, 0, 1])
        config = {"lambda_mmd": 0.01, "alpha_ce": 1.0, "training": {
            "epochs": 1, "learning_rate": 0.001, "weight_decay": 0.0,
            "min_delta": 0.0001, "patience": 2,
        }}
        with patch.object(proposal_mmd, "evaluate_ap", side_effect=[0.5, 0.6]), \
             patch.object(proposal_mmd, "mmd_loss", side_effect=lambda a, b: ((a.mean()-b.mean())**2, a.new_tensor(1.0))) as mmd:
            _, epoch, ap, history = proposal_mmd.train_mmd(model, [2, 2], [(x, y)], [x], [(x, y)], config)
        mmd.assert_called_once()
        self.assertEqual(epoch, 1)
        self.assertEqual(ap, 0.6)
        self.assertEqual([row["epoch"] for row in history], [0, 1])
        zero_paths = proposal_mmd.output_paths("unsw_to_cicids", 42, "configs/proposal_mmd_lambda0.json")
        main_paths = proposal_mmd.output_paths("unsw_to_cicids", 42, "configs/proposal_mmd_v1.json")
        self.assertNotEqual(zero_paths, main_paths)
        self.assertTrue(all("target_val_epoch0_" in str(path) for path in zero_paths + main_paths))


if __name__ == "__main__":
    unittest.main()
