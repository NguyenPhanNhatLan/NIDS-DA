import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from features.feature_pair_scoring import (
    ROOT, convert, distribution_similarity, greedy_select, load_training, run, score_pairs,
)
from features.common_features import transform_common_features


class FeaturePairScoringTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / 'configs/feature_selection_v1.json').read_text())
        self.config.update(sample_size=200, seeds=[42, 43, 44, 45, 46])
        self.pairs = json.loads((ROOT / 'configs/feature_pair_candidates.json').read_text())['pairs'][:2]
        rng = np.random.default_rng(9)
        self.unsw = pd.DataFrame({'dur': rng.exponential(size=400), 'spkts': rng.integers(1, 20, 400)})
        self.unsw['label'] = (self.unsw.dur > .7).astype(int)
        self.target = pd.DataFrame({'flow_duration': rng.exponential(size=400)*1e6,
                                    'total_fwd_packets': rng.integers(1, 20, 400),
                                    'label': rng.integers(0, 2, 400)})

    def score(self, target=None, pairs=None, config=None):
        return score_pairs(self.unsw, self.target if target is None else target,
                           self.pairs if pairs is None else pairs, self.config if config is None else config)

    def test_conversion_and_metrics(self):
        np.testing.assert_allclose(convert([1e6, 2e6], self.pairs[0], 'cicids'), [1, 2])
        x = np.arange(100.)
        ks, ws, wn = distribution_similarity(x, x)
        self.assertEqual((ks, ws, wn), (1., 1., 0.))
        self.assertLess(distribution_similarity(x, x+1000)[0], .1)
        self.assertTrue(np.isfinite(distribution_similarity(np.zeros(10), np.ones(10))[2]))

    def test_label_independence_and_determinism(self):
        a = self.score()
        b = self.score(self.target.drop(columns='label'))
        c = self.score(self.target.assign(label='poison'))
        d = self.score()
        for other in [b, c, d]:
            pd.testing.assert_frame_equal(a[0], other[0], check_exact=True)
            self.assertEqual(a[1:], other[1:])

    def test_reverse_target_label_independence(self):
        cfg = dict(self.config, source_domain='cicids')
        a = score_pairs(self.unsw, self.target, self.pairs, cfg)
        b = score_pairs(self.unsw.drop(columns='label'), self.target, self.pairs, cfg)
        self.assertEqual(a[1:], b[1:])

    def test_semantic_gate_and_missing(self):
        rejected = dict(self.pairs[0], canonical_name='rejected', semantic_valid=False)
        missing = dict(self.pairs[0], canonical_name='missing', cicids='absent')
        scores, selected, _ = self.score(pairs=self.pairs+[rejected, missing])
        self.assertNotIn('rejected', [r['canonical_name'] for r in selected])
        self.assertNotIn('missing', [r['canonical_name'] for r in selected])
        self.assertTrue(pd.isna(scores.set_index('canonical_name').loc['rejected', 'equivalence_score']))

    def test_quality_and_constant(self):
        target = self.target.copy()
        target['flow_duration'] = 0
        scores, selected, _ = self.score(target)
        self.assertEqual(len(selected), 1)
        self.assertEqual(scores.iloc[0]['status'], 'constant_or_empty')
        target = self.target.copy()
        target.loc[:20, 'flow_duration'] = np.inf
        target.loc[21:40, 'flow_duration'] = np.nan
        scores, _, _ = self.score(target)
        self.assertGreater(scores.iloc[0]['cicids_invalid_ratio'], 0)
        self.assertLess(scores.iloc[0]['quality_score'], 1)

    def test_redundancy(self):
        rows = [dict(canonical_name=n, equivalence_score=s, score_std=0.)
                for n,s in [('a', .9), ('duplicate', .89), ('independent', .8)]]
        corr = pd.DataFrame([[1,1,0],[1,1,0],[0,0,1]], index=['a','duplicate','independent'], columns=['a','duplicate','independent'])
        selected = greedy_select(rows, corr)
        self.assertEqual([r['canonical_name'] for r in selected], ['a','independent','duplicate'])
        self.assertAlmostEqual(selected[-1]['selection_gain'], .69)

    def test_parquet_projection_and_no_target_label(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for d, frame in [('unsw', self.unsw), ('cicids', self.target)]:
                folder = root / 'data/splits' / f'{d}_train'
                folder.mkdir(parents=True)
                frame.to_parquet(folder / 'part.parquet', index=False)
            from features import feature_pair_scoring as module
            real_read = module.pq.read_table
            with patch.object(module.pq, 'read_table', wraps=real_read) as read:
                frames, _ = load_training(self.config, self.pairs, root)
                for call in read.call_args_list:
                    if 'cicids_train' in str(call.args[0]):
                        self.assertNotIn('label', call.kwargs['columns'])
            self.assertNotIn('label', frames['cicids'])
            self.target.drop(columns='label').to_parquet(root / 'data/splits/cicids_train/part.parquet', index=False)
            frames2, _ = load_training(self.config, self.pairs, root)
            pd.testing.assert_frame_equal(frames['cicids'], frames2['cicids'])

    def test_aligned_transform_and_shortfall(self):
        _, selected, _ = self.score()
        artifact = dict(status='insufficient_eligible_pairs', pairs=self.pairs, features=selected)
        with self.assertRaises(ValueError):
            transform_common_features(self.target, artifact, 'cicids')
        transformed = transform_common_features(self.target, artifact, 'cicids', allow_shortfall=True)
        self.assertEqual(list(transformed), [r['canonical_name'] for r in selected])
        np.testing.assert_allclose(transformed.flow_duration, np.log1p(self.target.flow_duration / 1e6))

    def test_label_candidate_forbidden(self):
        with self.assertRaises(ValueError):
            self.score(pairs=[dict(self.pairs[0], cicids='label')])

    def test_cli_artifacts_for_all_k(self):
        # Synthetic compatible quantities exercise K > the local audit's pool.
        rng = np.random.default_rng(123)
        pairs = [dict(self.pairs[0], canonical_name=f'f{i}', unsw=f'u{i}',
                      cicids=f'c{i}', conversion='identity') for i in range(18)]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'configs').mkdir()
            cfg = dict(self.config, sample_size=50, seeds=[42, 43])
            for domain, prefix in [('unsw', 'u'), ('cicids', 'c')]:
                frame = pd.DataFrame({f'{prefix}{i}': rng.uniform(size=100) for i in range(18)})
                if domain == 'unsw':
                    frame['label'] = np.arange(100) % 2
                folder = root / 'data/splits' / f'{domain}_train'
                folder.mkdir(parents=True)
                frame.to_parquet(folder / 'part.parquet', index=False)
            (root / cfg['candidates']).write_text(json.dumps({'pairs': pairs}))
            config_path = root / 'configs/protocol.json'
            config_path.write_text(json.dumps(cfg))
            with contextlib.redirect_stdout(io.StringIO()):
                run(config_path)
            previous = []
            for k in cfg['top_k']:
                artifact = json.loads((root / f'configs/common_features_top{k}.json').read_text())
                self.assertEqual(artifact['actual_k'], k)
                self.assertEqual(artifact['status'], 'complete')
                names = [r['canonical_name'] for r in artifact['features']]
                self.assertEqual(names[:len(previous)], previous)
                previous = names
                self.assertEqual(len(pd.read_csv(root / f'results/feature_mapping/top{k}.csv')), k)


if __name__ == '__main__':
    unittest.main()
