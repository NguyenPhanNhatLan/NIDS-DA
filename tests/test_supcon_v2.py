from copy import deepcopy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from experiments import proposal_supcon_v2 as entry
from models.baseline import BaselineMLP
from models.contrastive_projection import ContrastiveProjection
from training.proposal_mkmmd import multi_kernel_mmd_loss
from training.supcon import supervised_contrastive_loss
from training.supcon_adaptation import effective_lambda, frozen_bn_training, initialize, losses, train


class SupConV2Tests(unittest.TestCase):
    def test_pairs_exclude_self_and_match_explicit_formula(self):
        features = torch.tensor([[1., 0.], [.8, .2], [0., 1.], [.2, .8]], dtype=torch.float64, requires_grad=True)
        labels = torch.tensor([0, 0, 1, 1])
        loss, stats = supervised_contrastive_loss(features, labels, .1)
        normalized = torch.nn.functional.normalize(features, dim=1)
        terms = []
        for i in range(4):
            candidates = [j for j in range(4) if j != i]
            positives = [j for j in candidates if labels[i] == labels[j]]
            denominator = torch.logsumexp(torch.stack([normalized[i] @ normalized[j] / .1 for j in candidates]), 0)
            terms.append(-torch.stack([normalized[i] @ normalized[j] / .1 - denominator for j in positives]).mean())
        torch.testing.assert_close(loss, torch.stack(terms).mean())
        self.assertEqual(stats['positive_pairs'], 4)
        self.assertEqual(stats['negative_pairs'], 8)
        self.assertEqual(stats['valid_anchors'], 4)
        loss.backward()
        self.assertTrue(torch.isfinite(features.grad).all())
        self.assertGreater(float(features.grad.abs().sum()), 0)

    def test_single_class_and_no_positive_batches_skip(self):
        for labels, reason in (([0, 0, 0], 'missing_source_class'), ([1, 1], 'missing_source_class'),
                               ([0, 1], 'no_positive_pairs'), ([0], 'missing_source_class')):
            features = torch.randn(len(labels), 64, requires_grad=True)
            loss, stats = supervised_contrastive_loss(features, torch.tensor(labels))
            self.assertEqual(loss.item(), 0)
            self.assertTrue(stats['skipped'])
            self.assertEqual(stats['skip_reason'], reason)
            loss.backward()
            torch.testing.assert_close(features.grad, torch.zeros_like(features))

    def test_extreme_values_stable_and_imbalanced_anchors_balanced(self):
        features = (torch.randn(10, 64) * 1e8).requires_grad_()
        labels = torch.tensor([0] * 8 + [1] * 2)
        loss, stats = supervised_contrastive_loss(features, labels, .001)
        self.assertEqual(stats['valid_anchors_per_class'], [8, 2])
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(features.grad).all())
        # Compute the unequal-count case explicitly to verify equal class weighting.
        z = torch.nn.functional.normalize(features.detach(), dim=1)
        terms = []
        for i in range(10):
            candidates = [j for j in range(10) if j != i]
            positives = [j for j in candidates if labels[i] == labels[j]]
            denominator = torch.logsumexp(torch.stack([z[i] @ z[j] / .001 for j in candidates]), 0)
            terms.append(-torch.stack([z[i] @ z[j] / .001 - denominator for j in positives]).mean())
        expected = (torch.stack(terms[:8]).mean() + torch.stack(terms[8:]).mean()) / 2
        torch.testing.assert_close(loss.detach(), expected)

    def test_projection_and_encoder_receive_gradients(self):
        torch.manual_seed(42)
        model, head = BaselineMLP(5), ContrastiveProjection()
        model.eval()
        z, logits = model(torch.randn(8, 5))
        projected = head(z)
        self.assertEqual(tuple(projected.shape), (8, 64))
        self.assertEqual(tuple(logits.shape), (8, 2))
        torch.testing.assert_close(projected.norm(dim=1), torch.ones(8))
        loss, _ = supervised_contrastive_loss(projected, torch.tensor([0] * 4 + [1] * 4))
        loss.backward()
        for parameter in (head.linear.weight, model.fc1.weight, model.fc2.weight):
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0)

    def test_independent_initialization_reproducible_and_classifier_needs_no_head(self):
        torch.manual_seed(11)
        original = deepcopy(BaselineMLP(5).state_dict())
        first, first_head = initialize(original, 5, 42, torch.device('cpu'))
        initial_head = deepcopy(first_head.state_dict())
        with torch.no_grad():
            first.fc1.weight.add_(5)
            first_head.linear.weight.add_(5)
        second, second_head = initialize(original, 5, 42, torch.device('cpu'))
        for key in original:
            torch.testing.assert_close(second.state_dict()[key], original[key])
        for key in initial_head:
            torch.testing.assert_close(second_head.state_dict()[key], initial_head[key])
        classification_only = BaselineMLP(5)
        classification_only.load_state_dict(second.state_dict(), strict=True)
        classification_only.eval()
        self.assertEqual(tuple(classification_only(torch.randn(4, 5))[1].shape), (4, 2))

    def test_warmup_can_be_disabled(self):
        config = entry.load_config(entry.CONFIGS['E2'], 'E2')
        self.assertAlmostEqual(effective_lambda(config, 1), .001)
        self.assertAlmostEqual(effective_lambda(config, 10), .01)
        self.assertAlmostEqual(effective_lambda(config, 20), .01)
        config['mmd_warmup_epochs'] = 0
        self.assertAlmostEqual(effective_lambda(config, 1), .01)

    def test_fixed_seed_one_step_is_reproducible(self):
        torch.manual_seed(9)
        source_state = deepcopy(BaselineMLP(5).state_dict())
        sx, tx = torch.randn(8, 5), torch.randn(8, 5)
        y = torch.tensor([0] * 4 + [1] * 4)
        config = entry.load_config(entry.CONFIGS['E2'], 'E2')
        observations = []
        for _ in range(2):
            model, head = initialize(source_state, 5, 42, torch.device('cpu'))
            frozen_bn_training(model)
            optimizer = torch.optim.Adam([*model.parameters(), *head.parameters()], lr=.001)
            total, _, _ = losses(model, head, sx, y, tx, nn.CrossEntropyLoss(), config, 1)
            total.backward()
            optimizer.step()
            observations.append((float(total.detach()), deepcopy(model.state_dict()), deepcopy(head.state_dict())))
        self.assertEqual(observations[0][0], observations[1][0])
        for position in (1, 2):
            for key in observations[0][position]:
                torch.testing.assert_close(observations[0][position][key], observations[1][position][key])

    def test_supcon_disabled_preserves_existing_mkmmd_formula_and_gradients(self):
        config = entry.load_config(entry.CONFIGS['E2'], 'E2')
        config.update(alpha_supcon=0, mmd_warmup_epochs=0)
        torch.manual_seed(7)
        model, head = BaselineMLP(5), ContrastiveProjection()
        model.eval()
        sx, tx = torch.randn(8, 5), torch.randn(8, 5)
        y = torch.tensor([0] * 4 + [1] * 4)
        criterion = nn.CrossEntropyLoss()
        value, _, _ = losses(model, head, sx, y, tx, criterion, config, 1)
        z, logits = model(torch.cat([sx, tx]))
        expected = criterion(logits[:8], y) + .01 * multi_kernel_mmd_loss(z[:8], z[8:], config['kernel_scales'])[0]
        torch.testing.assert_close(value, expected)
        actual_grad = torch.autograd.grad(value, model.fc1.weight, retain_graph=True)[0]
        expected_grad = torch.autograd.grad(expected, model.fc1.weight)[0]
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-4, atol=1e-6)

    def test_e1_does_not_iterate_target_and_e2_rejects_target_labels(self):
        class ForbiddenTarget:
            def __iter__(self):
                raise AssertionError('E1 accessed target training')
        config = entry.load_config(entry.CONFIGS['E1'], 'E1')
        config['training'].update(epochs=1)
        torch.manual_seed(42)
        model, head = BaselineMLP(5), ContrastiveProjection()
        samples = [(torch.randn(8, 5), torch.tensor([0] * 4 + [1] * 4))]
        metrics = {'pr_auc': .5, 'roc_auc': .5}
        with patch('training.supcon_adaptation.source_evaluation', return_value=metrics):
            train(model, head, samples, ForbiddenTarget(), samples, [4, 4], config, .5)
        config = entry.load_config(entry.CONFIGS['E2'], 'E2')
        config['training'].update(epochs=1)
        with patch('training.supcon_adaptation.source_evaluation', return_value=metrics):
            with self.assertRaisesRegex(ValueError, 'features only'):
                train(model, head, samples, samples, samples, [4, 4], config, .5)

    def test_parquet_one_batch_cpu_smoke(self):
        from experiments.supcon_smoke import smoke
        result = smoke(torch.device('cpu'))
        self.assertEqual(set(result), {'E1', 'E2'})
        self.assertTrue(all(row['status'] == 'passed' for row in result.values()))

    def test_synthetic_diagnostics_cover_all_three_checkpoints(self):
        from evaluation import proposal_negative_transfer as legacy_diagnostics
        from evaluation import supcon_diagnostics as diagnostics
        rng = np.random.default_rng(42)
        sx = rng.normal(size=(32, 5)).astype(np.float32)
        tx = (rng.normal(size=(32, 5)) + .3).astype(np.float32)
        labels = np.array([0] * 16 + [1] * 16, dtype=np.int64)
        torch.manual_seed(42)
        models = {name: BaselineMLP(5) for name in ('source_only', 'E1', 'E2')}
        heads = {name: ContrastiveProjection() for name in ('E1', 'E2')}
        configs = {exp: entry.load_config(entry.CONFIGS[exp], exp) for exp in ('E1', 'E2')}
        for config in configs.values():
            config['training']['batch_size'] = 8
            config['diagnostics'].update(sample_rows=32, mmd_sample=8, mmd_repeats=2, cv_folds=2, gradient_batches=1)
        paths = {name: Path(name) for name in ('source_train', 'source_val', 'target_train', 'target_val')}

        def samples(path, *args):
            return (sx, labels) if path.name == 'source_val' else (tx, labels)

        def stream(path, *args, **kwargs):
            if path.name == 'source_train':
                return [(torch.from_numpy(sx[:8]), torch.tensor([0] * 4 + [1] * 4))]
            return [torch.from_numpy(tx[:8])]

        with patch.object(legacy_diagnostics, 'sample_labeled', side_effect=samples), \
             patch.object(diagnostics, 'ParquetBatchStream', side_effect=stream):
            report = diagnostics.diagnose(models, heads, paths, [16, 16], configs, 42)
        self.assertEqual(set(report['runs']), {'source_only', 'E1', 'E2'})
        self.assertEqual(len(report['fixed_bandwidth_mmd']['paired_indices']), 2)
        for exp in ('E1', 'E2'):
            self.assertTrue(np.isfinite(report['runs'][exp]['domain_auc']['auc_mean']))
            self.assertIn('separation_ratio', report['runs'][exp]['class_conditional'])
            self.assertEqual(report['runs'][exp]['gradient_cosine']['ce_supcon_vs_mkmmd']['batches'], 1)

    def test_lock_refusal_precedes_loading_or_writing(self):
        with patch.object(entry, 'require_development_open', side_effect=RuntimeError('Development is locked')), \
             patch.object(entry, 'reference') as reference:
            with self.assertRaisesRegex(RuntimeError, 'locked'):
                entry.run_train('E1', 'unsw_to_cicids', 42, entry.CONFIGS['E1'], 'cpu')
            reference.assert_not_called()

    def test_existing_artifact_is_preserved_and_final_output_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            existing_model, result = root / 'seed42.pt', root / 'seed42.json'
            final = root / 'final_test/seed42.json'
            final.parent.mkdir()
            existing_model.write_bytes(b'original checkpoint')
            final.write_text('original final result')
            with patch.object(entry, 'require_development_open'), \
                 patch.object(entry, 'artifact_paths', return_value=(existing_model, result)), \
                 patch.object(entry, 'reference') as reference:
                with self.assertRaises(FileExistsError):
                    entry.run_train('E1', 'unsw_to_cicids', 42, entry.CONFIGS['E1'], 'cpu')
                reference.assert_not_called()
            self.assertEqual(existing_model.read_bytes(), b'original checkpoint')
            self.assertEqual(final.read_text(), 'original final result')
            self.assertFalse(result.exists())

    def test_frozen_code_files_not_modified_by_smoke(self):
        from experiments.supcon_smoke import smoke
        files = [entry.ROOT / path for path in ('src/models/baseline.py', 'src/training/proposal_mmd.py',
                 'src/training/proposal_mkmmd.py', 'src/training/proposal_class_aware.py',
                 'src/training/proposal_data.py', 'src/experiments/proposal_pipeline.py', 'docs/thesis_protocol.md')]
        hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
        with patch.object(entry, 'reference', side_effect=AssertionError('Real data accessed')):
            smoke(torch.device('cpu'))
        self.assertEqual(hashes, {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in files})


if __name__ == '__main__':
    unittest.main()
