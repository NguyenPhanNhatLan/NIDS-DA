import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from evaluation.proposal_data_audit import inventory
from features import freeze_proposal_revision as freeze_module
from features.common_features import COMMON_FEATURES
from training.data_revision import verify_revision


class FrozenRevisionTests(unittest.TestCase):
    def test_relocation_audit_reports_labels_and_enforces_design_limits(self):
        initial = {d: np.array([0, 0, 1, 1, 2, 2]) for d in freeze_module.DOMAINS}
        final = {d: np.array([0, 2, 1, 1, 2, 2]) for d in freeze_module.DOMAINS}
        labels = {d: np.array([0, 1, 0, 1, 0, 1]) for d in freeze_module.DOMAINS}
        report = freeze_module.relocation_audit(initial, final, labels, .01, .005)
        self.assertFalse(report['quality_gate_passed'])
        domain = report['domains']['unsw']
        self.assertEqual(domain['relocated_unique_rows'], 1)
        self.assertEqual(domain['transitions']['train_to_test']['class_counts'], [0, 1])
        self.assertEqual(domain['initial']['train']['rows'], 2)
        self.assertEqual(domain['final']['train']['rows'], 1)
        self.assertEqual(domain['attack_prevalence_change']['train'], -.5)

    def test_freeze_preserves_rows_groups_and_rejects_mutations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'configs').mkdir()
            (root / 'configs/common_features_v2.json').write_text('[]')
            origin = root / 'origin'
            origin.mkdir()
            source_manifest = origin / 'manifest.json'
            source_manifest.write_text(json.dumps({'raw_inputs': {}}))
            audit = {'quality_gate_passed': True, 'source_manifest': {
                'path': str(source_manifest), 'sha256': freeze_module.file_sha256(source_manifest)},
                'common_inventory': {}}
            for domain in freeze_module.DOMAINS:
                audit['common_inventory'][domain] = {}
                for index, split in enumerate(freeze_module.SPLITS):
                    path = origin / f'{domain}_{split}'
                    path.mkdir()
                    x = (np.arange(100).reshape(20, 5) + index * 200).astype(float)
                    if split in ('train', 'val'):
                        x[0] = [1. + index * 1e-12, 2, 3, 4, 5]
                    table = pa.table({**{n: x[:, i] for i, n in enumerate(COMMON_FEATURES)},
                                      'label': np.arange(20) % 2})
                    pq.write_table(table, path / 'part.parquet')
                    audit['common_inventory'][domain][split] = inventory(path)
            audit_path = root / 'audit.json'
            audit_path.write_text(json.dumps(audit))
            with patch.object(freeze_module, 'ROOT', root):
                manifest_path = freeze_module.freeze(audit_path, 'fixture', max_relocated_fraction=.1, max_class_prevalence_change=.1)
            manifest = json.loads(manifest_path.read_text())
            for direction in ('unsw_to_cicids', 'cicids_to_unsw'):
                for domain in freeze_module.DOMAINS:
                    self.assertEqual(sum(n for key, n in manifest['prepared_counts'][direction].items()
                                         if key.startswith(domain)), 60)
            quality = json.loads(Path(manifest['prepared_quality']).read_text())
            self.assertTrue(all(q['cross_split_vector_groups'] == 0 for d in quality.values() for q in d.values()))
            self.assertGreater(manifest['precision_relocation_iterations'][0]['unsw']['train'], 0)
            with patch.dict(os.environ, {'KLTN_DATA_MANIFEST': str(manifest_path)}):
                self.assertEqual(verify_revision()['data_revision'], 'fixture')
                file = Path(manifest['feature_root']) / 'unsw_to_cicids/unsw_train/extra.parquet'
                pq.write_table(table, file)
                with self.assertRaisesRegex(ValueError, 'inventory changed'):
                    verify_revision()
                file.unlink()
                (root / 'configs/common_features_v2.json').write_text('[1]')
                with self.assertRaisesRegex(ValueError, 'changed'):
                    verify_revision()


if __name__ == '__main__':
    unittest.main()
