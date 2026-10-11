"""Synthetic checks for the extension; no canonical dataset or training runs."""
from copy import deepcopy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from training.supcon_extension import supervised_contrastive_loss
from experiments import proposal_supcon_extension as extension


class SupConLossTests(unittest.TestCase):
    def test_matches_explicit_positive_pair_formula(self):
        features = torch.tensor([[1., 0.], [.8, .2], [0., 1.], [.2, .8]], requires_grad=True)
        labels = torch.tensor([0, 0, 1, 1])
        temperature = .2
        loss, anchors = supervised_contrastive_loss(features, labels, temperature)
        normalized = torch.nn.functional.normalize(features, dim=1)
        terms = []
        for i in range(4):
            candidates = [j for j in range(4) if j != i]
            positives = [j for j in candidates if labels[j] == labels[i]]
            denominator = sum(torch.exp(normalized[i] @ normalized[j] / temperature) for j in candidates)
            terms.append(-sum(torch.log(torch.exp(normalized[i] @ normalized[j] / temperature) / denominator)
                              for j in positives) / len(positives))
        self.assertEqual(anchors, 4)
        torch.testing.assert_close(loss, torch.stack(terms).mean())
        loss.backward()
        self.assertTrue(torch.isfinite(features.grad).all())

    def test_no_positive_returns_differentiable_zero(self):
        features = torch.randn(2, 3, requires_grad=True)
        loss, anchors = supervised_contrastive_loss(features, torch.tensor([0, 1]))
        self.assertEqual(anchors, 0)
        self.assertEqual(loss.item(), 0)
        loss.backward()
        torch.testing.assert_close(features.grad, torch.zeros_like(features))

    def test_permutation_invariant_and_ignores_singleton_anchor(self):
        features = torch.tensor([[1., 0.], [.7, .3], [0., 1.]])
        labels = torch.tensor([0, 0, 1])
        value, anchors = supervised_contrastive_loss(features, labels)
        order = torch.tensor([2, 0, 1])
        other, count = supervised_contrastive_loss(features[order], labels[order])
        self.assertEqual((anchors, count), (2, 2))
        torch.testing.assert_close(value, other)


class ExtensionInitializationTests(unittest.TestCase):
    def test_e1_e2_load_same_source_even_after_e1_saved(self):
        config = extension.load_config(extension.DEFAULT_CONFIG)
        torch.manual_seed(7)
        model = extension.BaselineMLP(len(extension.COMMON_FEATURES))
        source_state = deepcopy(model.state_dict())
        initialized = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / 'source.pt'
            source_result = root / 'source.json'
            torch.save({'model_state_dict': source_state}, source_path)
            source_result.write_text('{}')
            validated = {'revision': {'data_revision': 'fixture', 'data_manifest_sha256': 'fixture'},
                         'checkpoint_path': source_path, 'result_path': source_result,
                         'base': root, 'target': 'cicids', 'source_val_ap': .5}
            checkpoint = {'model_state_dict': source_state, 'prepared_split_sha256': {},
                          'preprocessor_sha256': 'fixture', 'common_feature_config_sha256': 'fixture'}
            baseline = {'target_development': {key: .5 for key in extension.METRICS}}

            def locations(direction, seed, experiment):
                return root / experiment / 'model.pt', root / experiment / 'result.json'

            def stream(*args, **kwargs):
                return [(torch.ones(4, len(extension.COMMON_FEATURES)), torch.tensor([0, 0, 1, 1]))]

            def fake_train(model, source, target, validation, counts, config, experiment, initial_ap):
                initialized.append((experiment, deepcopy(model.state_dict()), target))
                with torch.no_grad():
                    model.fc1.weight.add_(10)
                return model, 1, .5, []

            with patch.object(extension, 'require_development_open'), \
                 patch.object(extension, 'verify_revision', return_value=validated['revision']), \
                 patch.object(extension, 'reference', return_value=(validated, checkpoint, baseline)), \
                 patch.object(extension, 'artifact_paths', side_effect=locations), \
                 patch.object(extension, 'ParquetBatchStream', side_effect=stream), \
                 patch.object(extension, 'train_extension', side_effect=fake_train), \
                 patch.object(extension, 'collect_scores', return_value=([0, 1], [.2, .8])), \
                 patch.object(extension, 'select_f1_threshold', return_value=.5), \
                 patch.object(extension, 'compute_metrics', return_value=baseline['target_development']):
                extension.train('unsw_to_cicids', 42, 'E1', extension.DEFAULT_CONFIG)
                extension.train('unsw_to_cicids', 42, 'E2', extension.DEFAULT_CONFIG)
                with self.assertRaises(FileExistsError):
                    extension.train('unsw_to_cicids', 42, 'E1', extension.DEFAULT_CONFIG)
            self.assertEqual([row[0] for row in initialized], ['E1', 'E2'])
            self.assertIsNone(initialized[0][2])
            self.assertIsNotNone(initialized[1][2])
            for _, state, _ in initialized:
                for key in source_state:
                    torch.testing.assert_close(state[key], source_state[key])


if __name__ == '__main__':
    unittest.main()
