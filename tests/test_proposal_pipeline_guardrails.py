"""Pure-stdlib regression tests for canonical row counts and safe train resume."""
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments import proposal_pipeline as pipeline


class ProposalPipelineGuardrailsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'configs').mkdir()
        (self.root / 'configs' / 'proposal_mmd_v2.json').write_text(
            json.dumps({'method': 'marginal_mmd'})
        )
        self.direction = 'unsw_to_cicids'
        self.seed = 42
        self.common = self.root / 'common'
        self.prepared = self.root / 'prepared'
        patches = [
            patch.object(pipeline, 'ROOT', self.root),
            patch.object(pipeline, 'DIRECTIONS', (self.direction,)),
            patch.object(pipeline, 'SEEDS', (self.seed,)),
            patch.object(pipeline, 'CONFIGS', ('proposal_mmd_v2',)),
            patch.object(pipeline, 'SUITE', {'methods': ['source_only', 'marginal_mmd']}),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

        # Metadata-only fixture; no actual Parquet readers or PySpark are needed.
        for d in ('unsw', 'cicids'):
            for split in ('train', 'val', 'test'):
                for root in (self.common, self.prepared / self.direction):
                    folder = root / f'{d}_{split}'
                    folder.mkdir(parents=True)
                    (folder / 'part.parquet').write_text('12')

        parquet = types.ModuleType('pyarrow.parquet')
        parquet.ParquetFile = lambda path: types.SimpleNamespace(
            metadata=types.SimpleNamespace(num_rows=int(Path(path).read_text()))
        )
        arrow = types.ModuleType('pyarrow')
        arrow.__path__ = []
        arrow.parquet = parquet
        revision = types.ModuleType('training.data_revision')
        revision.revision_path = lambda kind, fallback: {
            'common_root': self.common, 'feature_root': self.prepared
        }[kind]
        revision.require_development_open = lambda: None
        self.revision = revision
        self.patcher = patch.dict(sys.modules, {
            'pyarrow': arrow, 'pyarrow.parquet': parquet,
            'training.data_revision': revision,
        })
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def fake_experiment_modules(self, events):
        def locations(method, direction, seed):
            root = self.root / 'models' / method / direction
            return root / f'seed{seed}.pt', root / f'seed{seed}.json'

        def create(method, direction, seed):
            checkpoint, result = locations(method, direction, seed)
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_text('model')
            result.write_text('result')
            events.append(('created', method))

        baseline = types.ModuleType('experiments.proposal_source_only')
        baseline.run = lambda direction, seed: create('source_only', direction, seed)
        mmd = types.ModuleType('training.proposal_mmd')
        mmd.run = lambda direction, seed, config: create('marginal_mmd', direction, seed)
        final = types.ModuleType('evaluation.proposal_final_test')
        final.paths = locations
        final.validate_development = lambda direction, method, seed: events.append(('validated', method))
        return patch.dict(sys.modules, {
            'experiments.proposal_source_only': baseline,
            'training.proposal_mmd': mmd,
            'evaluation.proposal_final_test': final,
        }), locations

    def test_counts_match_and_mismatch_fail_closed(self):
        pipeline.verify_input_row_counts()
        bad = self.prepared / self.direction / 'cicids_train' / 'part.parquet'
        bad.write_text('11')
        with self.assertRaisesRegex(ValueError, 'Inconsistent inputs'):
            pipeline.verify_input_row_counts()

    def test_resume_validates_existing_before_creating_missing(self):
        events = []
        patches, locations = self.fake_experiment_modules(events)
        with patches:
            checkpoint, result = locations('source_only', self.direction, self.seed)
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_text('old model')
            result.write_text('old result')
            pipeline.resume_train()
        self.assertEqual(events, [
            ('validated', 'source_only'), ('created', 'marginal_mmd')
        ])

    def test_resume_rejects_half_written_pair(self):
        events = []
        patches, locations = self.fake_experiment_modules(events)
        with patches:
            checkpoint, _ = locations('source_only', self.direction, self.seed)
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_text('interrupted')
            with self.assertRaisesRegex(RuntimeError, 'Incomplete source_only artifacts'):
                pipeline.resume_train()
        self.assertEqual(events, [])

    def test_resume_obeys_development_lock(self):
        events = []
        patches, _ = self.fake_experiment_modules(events)
        self.revision.require_development_open = lambda: (_ for _ in ()).throw(
            RuntimeError('Development is locked')
        )
        with patches, self.assertRaisesRegex(RuntimeError, 'Development is locked'):
            pipeline.resume_train()
        self.assertEqual(events, [])


if __name__ == '__main__':
    unittest.main()
