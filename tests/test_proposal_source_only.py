import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from experiments import proposal_source_only as experiment


class ProposalSourceOnlyTests(unittest.TestCase):
    def test_domains_and_counts(self):
        self.assertEqual(experiment.domains("unsw_to_cicids"), ("unsw", "cicids"))
        self.assertEqual(experiment.domains("cicids_to_unsw"), ("cicids", "unsw"))
        with self.assertRaises(ValueError):
            experiment.domains("other")
        batches = [(None, torch.tensor([0, 1, 1])), (None, torch.tensor([0]))]
        self.assertEqual(experiment.count_classes(batches), [2, 2])

    def test_source_validation_selects_threshold(self):
        loaders = []

        def fake_loader(path, batch_size, shuffle, seed, include_labels, drop_last=False):
            token = (path.name, batch_size, shuffle, seed, include_labels, drop_last)
            loaders.append(token)
            return token

        class FakeModel:
            def __init__(self, input_dim):
                self.input_dim = input_dim

            def to(self, device):
                return self

            def state_dict(self):
                return {"weight": torch.tensor([1.0])}

        def fake_scores(model, loader):
            if loader[0] == "unsw_val":
                return np.array([0, 1]), np.array([0.1, 0.9])
            return np.array([0, 1]), np.array([0.2, 0.8])

        with tempfile.TemporaryDirectory() as temp_dir, \
             patch.object(experiment, "CHECKPOINT_ROOT", Path(temp_dir) / "models"), \
             patch.object(experiment, "RESULT_ROOT", Path(temp_dir) / "results"), \
             patch.object(experiment, "ParquetBatchStream", side_effect=fake_loader), \
             patch.object(experiment, "split_sha256", return_value="fixture-hash"), \
             patch.object(experiment, "count_classes", return_value=[2, 2]) as counts, \
             patch.object(experiment, "BaselineMLP", FakeModel), \
             patch.object(experiment, "train_baseline", side_effect=lambda model, *a, **k: (model, 3, 0.8)), \
             patch.object(experiment, "collect_scores", side_effect=fake_scores), \
             patch.object(experiment, "select_f1_threshold", return_value=0.6) as select, \
             patch.object(experiment, "compute_metrics", return_value={"f1": 1.0}):
            result = experiment.run("unsw_to_cicids", 42)
            checkpoint = torch.load(result["checkpoint"], map_location="cpu", weights_only=True)
            self.assertEqual(checkpoint["best_epoch"], 3)
            self.assertEqual(checkpoint["direction"], "unsw_to_cicids")
        self.assertEqual(loaders[0], ("unsw_train", 4096, False, 42, True, False))
        self.assertEqual(loaders[1], ("unsw_train", 256, True, 42, True, True))
        self.assertEqual(loaders[2][0], "unsw_val")
        self.assertEqual(loaders[3][0], "cicids_val")
        counts.assert_called_once_with(loaders[0])
        select.assert_called_once()
        self.assertEqual(result["threshold_from_source_val"], 0.6)


if __name__ == "__main__":
    unittest.main()
