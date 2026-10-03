import unittest
from unittest.mock import patch

import torch

from models.baseline import BaselineMLP
from training import proposal_mmd


class ProposalMmdPretrainedTests(unittest.TestCase):
    def test_lambda_zero_is_ce_only_and_bn_stats_stay_frozen(self):
        model = BaselineMLP(10)
        initial_state = {key: value.clone() for key, value in model.state_dict().items()}
        before = [(m.running_mean.clone(), m.running_var.clone())
                  for m in model.modules() if isinstance(m, torch.nn.BatchNorm1d)]
        x = torch.randn(4, 10)
        y = torch.tensor([0, 1, 0, 1])
        config = {"lambda_mmd": 0.0, "training": {
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
        config = {"lambda_mmd": 0.01, "training": {
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
        self.assertTrue(all("pretrained_" in str(path) for path in zero_paths + main_paths))


if __name__ == "__main__":
    unittest.main()
