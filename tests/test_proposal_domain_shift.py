import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from evaluation import proposal_domain_shift as diagnostic


class IntrinsicShiftTests(unittest.TestCase):
    def test_canonical_scalar_training_sample_is_reproducible(self):
        with tempfile.TemporaryDirectory() as directory:
            x = np.arange(200, dtype=float).reshape(40, 5)
            table = pa.table({name: x[:, i] for i, name in enumerate(diagnostic.COMMON_FEATURES)})
            pq.write_table(table, Path(directory) / 'part.parquet')
            first, total, rows = diagnostic.sample_features(
                directory, 20, 42, allow_nonfinite=True, return_indices=True)
            second, _, again = diagnostic.sample_features(
                directory, 20, 42, allow_nonfinite=True, return_indices=True)
            self.assertEqual(total, 40)
            np.testing.assert_array_equal(rows, again)
            np.testing.assert_array_equal(first, second)
            np.testing.assert_array_equal(first, x[rows])

    def test_fixed_pairs_symmetry_and_controls(self):
        rng = np.random.default_rng(12)
        u = rng.normal(size=(80, 5))
        c = rng.normal(1, 2, size=(100, 5))
        result, pairs = diagnostic.intrinsic_mmd(u, c, 16, 3, 42)
        again, same_pairs = diagnostic.intrinsic_mmd(u, c, 16, 3, 42)
        self.assertEqual(result, again)
        for key in pairs:
            np.testing.assert_array_equal(pairs[key], same_pairs[key])
            self.assertEqual(len(np.unique(pairs[key])), 32)
        for record in result['runs']:
            self.assertLess(record['absolute_symmetry_error'], result['symmetry_epsilon'])
            self.assertAlmostEqual(record['identical_sample_mmd2'], 0, places=14)
            self.assertGreaterEqual(record['unsw_same_domain'], -result['symmetry_epsilon'])
        self.assertIn('not a 95%', result['std_interpretation'])

    def test_auc_duplicate_vectors_never_cross_holdout(self):
        rng = np.random.default_rng(11)
        u = np.repeat(rng.normal(size=(100, 5)), 3, axis=0)
        c = np.repeat(rng.normal(3, 1, size=(100, 5)), 3, axis=0)
        result, indices = diagnostic.intrinsic_auc(u, c)
        x = np.concatenate([u, c])
        train = {tuple(row) for row in x[indices['train']]}
        test = {tuple(row) for row in x[indices['test']]}
        self.assertFalse(train & test)
        self.assertEqual(result['duplicate_groups_overlap'], 0)
        self.assertGreater(result['auc'], .9)
        self.assertTrue(all(n > 0 for n in result['test_domain_counts']))

    def test_pipeline_artifacts_and_output_guard(self):
        rng = np.random.default_rng(1)

        def sample(path, count, seed, **kwargs):
            return rng.lognormal(size=(100, 5)), 100, np.arange(100)

        with tempfile.TemporaryDirectory() as directory, \
                patch.object(diagnostic, 'RESULT_ROOT', Path(directory)), \
                patch.object(diagnostic, 'sample_features', side_effect=sample), \
                patch.object(diagnostic, 'parquet_files', return_value=[]):
            result = diagnostic.run_intrinsic(Path(directory), mmd_sample=8, mmd_repeats=2)
            self.assertFalse(result['intrusion_labels_used'])
            with np.load(result['sample_artifact']) as artifacts:
                self.assertIn('unsw_selected_global_rows', artifacts)
                self.assertIn('cicids_repeat0', artifacts)
            with self.assertRaises(FileExistsError):
                diagnostic.run_intrinsic(Path(directory))


if __name__ == '__main__':
    unittest.main()
