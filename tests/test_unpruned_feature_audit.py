import json
from pathlib import Path
import shutil
import tempfile
import unittest

import pandas as pd

from features.feature_pair_scoring import load_training, score_pairs
from features.unpruned_feature_audit import verify_freeze
from features.unpruned_audit_data import ROOT


class FrozenAuditTests(unittest.TestCase):
    def test_semantic_gate_unchanged_and_recovered_features_available(self):
        baseline = json.loads((ROOT / 'configs/feature_pair_candidates.json').read_text())['pairs']
        expanded = json.loads((ROOT / 'configs/feature_pair_candidates_unpruned_v1.json').read_text())['pairs']
        self.assertEqual({p['canonical_name']:p['semantic_valid'] for p in baseline},
                         {p['canonical_name']:p['semantic_valid'] for p in expanded})
        table = pd.read_csv(ROOT / 'results/feature_mapping/unpruned_v1/semantic_audit.csv')
        self.assertTrue(table.available_cicids.all())
        self.assertEqual(int(table.semantic_valid.sum()), 3)

    def test_frozen_decisions_and_config_cannot_change(self):
        name = 'configs/feature_selection_unpruned_v1.json'
        cfg = json.loads((ROOT / name).read_text())
        freeze = json.loads((ROOT / cfg['semantic_freeze']).read_text())
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for filename in [*freeze['files'], cfg['semantic_freeze']]:
                target = root / filename
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / filename, target)
            (root / cfg['dataset_manifest']).parent.joinpath('_SUCCESS').touch()
            verify_freeze(root / name)
            (root / cfg['candidates']).write_text('{}')
            with self.assertRaisesRegex(ValueError, 'Frozen audit file changed'):
                verify_freeze(root / name)

    def test_projected_order_independent_of_physical_row_order(self):
        cfg = json.loads((ROOT / 'configs/feature_selection_v1.json').read_text())
        cfg.update(deterministic_row_order=True, sample_size=4, seeds=[42])
        pairs = json.loads((ROOT / 'configs/feature_pair_candidates.json').read_text())['pairs'][:2]
        source = pd.DataFrame(dict(dur=[4.,1.,3.,2.], spkts=[4,1,3,2], label=[1,0,1,0]))
        target = pd.DataFrame(dict(flow_duration=[4e6,1e6,3e6,2e6], total_fwd_packets=[4,1,3,2]))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for d, frame in [('unsw',source),('cicids',target)]:
                folder = root / cfg['training_paths'][d]
                folder.mkdir(parents=True)
                frame.to_parquet(folder / 'part.parquet', index=False)
            first, manifest1 = load_training(cfg, pairs, root)
            for d, frame in [('unsw',source),('cicids',target)]:
                frame.iloc[::-1].to_parquet(root / cfg['training_paths'][d] / 'part.parquet', index=False)
            second, manifest2 = load_training(cfg, pairs, root)
            for d in first:
                pd.testing.assert_frame_equal(first[d], second[d])
                self.assertEqual(manifest1[d]['projected_data_sha256'], manifest2[d]['projected_data_sha256'])
            self.assertNotIn('label', second['cicids'])


if __name__ == '__main__':
    unittest.main()
