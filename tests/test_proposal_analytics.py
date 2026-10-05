import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import pyarrow.parquet as pq

from analytics.build_dashboard_tables import FEATURES, METRICS, SCHEMAS, build_tables, export


class DashboardTablesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.results = self.root / 'results' / 'proposal_v2'
        self.results.mkdir(parents=True)
        self.direction = 'unsw_to_cicids'
        self.shared = dict(direction=self.direction, seed=42, features=FEATURES, feature_count=5,
                           preprocessor_sha256='processor', common_feature_config_sha256='schema',
                           prepared_split_sha256={'source_train': 'a', 'source_val': 'b', 'target_val': 'c'})

    def write(self, rel, data):
        path = self.results / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
        return path

    def metrics(self, ap):
        return dict(pr_auc=ap, roc_auc=.7, macro_f1=.5, recall=.6, fpr=.1, tn=8, fp=2, fn=4, tp=6)

    def fixtures(self):
        baseline = {**self.shared, 'target_development': self.metrics(.2), 'source_val': self.metrics(.5),
                    'source_train_counts': [80, 20], 'checkpoint': 'source.pt'}
        adapted = {**self.shared, 'protocol': 'proposal_v2', 'method': 'marginal_mmd',
                   'cross_domain': self.metrics(.3), 'within_domain': self.metrics(.6), 'lambda_mmd': .01,
                   'config_sha256': 'config', 'source_checkpoint': 'source.pt', 'checkpoint': 'adapted.pt'}
        self.write(f'source_only_target_val/{self.direction}/seed42.json', baseline)
        self.write(f'mmd/target_val_epoch0_proposal_mmd_v2/{self.direction}/seed42.json', adapted)
        diagnostic = {'direction': self.direction, 'seed': 42, 'config': 'configs/proposal_mmd_v2.json',
            'config_sha256': 'config', 'artifacts': {'adapted_checkpoint': 'adapted.pt'},
            'negative_transfer': {'detected': False, 'delta_target_pr_auc': .1},
            'marginal_alignment': {'before': {'mmd2_mean': .5}, 'after': {'mmd2_mean': .2}},
            'domain_separability': {'before': {'auc_mean': .9}, 'after': {'auc_mean': .8}},
            'gradient_conflict': {'mean': -.1, 'negative_fraction': .7}, 'label_prior': {'absolute_gap': .2},
            'likely_causes': ['ce_mmd_gradient_conflict']}
        diag = self.write(f'diagnostics_target_val/{self.direction}/seed42.json', diagnostic)
        self.write('domain_shift/shift.json', {**self.shared, 'protocol': 'proposal_v2',
            'ks_by_feature': {name: {'ks_statistic': .4, 'p_value': .01, 'source_median': 1, 'target_median': 2} for name in FEATURES},
            'input_mmd': {'mmd2_mean': .6}, 'domain_classifier': {'auc_mean': .95},
            'sampling': {'source_total_rows': 100, 'target_total_rows': 200}})
        for method, ap, cp in [('source_only', .4, 'source.pt'), ('marginal_mmd', .6, 'adapted.pt')]:
            self.write(f'final_target_test/{method}/final.json', {**self.shared, 'protocol': 'proposal_v2',
                'method': method, 'target_test': self.metrics(ap), 'target_test_sha256': 'test', 'checkpoint': cp})
        per_run = [{**self.shared, 'method': method, **self.metrics(ap), 'adaptation_gain_pr_auc': gain}
                   for method, ap, gain in [('source_only', .2, 0), ('marginal_mmd', .3, .1)]]
        summary = []
        for row in per_run:
            stats = {key: {'mean': row[key], 'std': 0., 'ci95': None} for key in METRICS + ['adaptation_gain_pr_auc']}
            summary.append({'direction': self.direction, 'method': row['method'], 'seeds': [42], 'metrics': stats})
        self.write('aggregate/summary.json', {'protocol': 'proposal_v2', 'data_role': 'development_only',
                                            'per_run': per_run, 'summary': summary})
        return diag

    def test_all_artifact_families_and_phase_safe_joins(self):
        self.fixtures()
        output = self.root / 'analytics'
        manifest = export(self.results, output)
        self.assertEqual(set(SCHEMAS), {p.stem for p in output.glob('*.parquet')})
        self.assertEqual(manifest['row_counts']['experiment_runs'], 4)  # aggregate does not double-count runs
        self.assertEqual(manifest['row_counts']['feature_shift'], 5)
        rows = pq.read_table(output / 'experiment_runs.parquet').to_pylist()
        development = next(row for row in rows if row['method'] == 'marginal_mmd' and row['phase'] == 'development')
        final = next(row for row in rows if row['method'] == 'marginal_mmd' and row['phase'] == 'final_test')
        self.assertAlmostEqual(development['adaptation_gain'], .1)
        self.assertEqual(development['mmd_before'], .5)
        self.assertAlmostEqual(final['adaptation_gain'], .2)
        self.assertIsNone(final['mmd_before'])
        self.assertEqual(final['lambda'], .01)
        self.assertEqual(pq.read_table(output / 'negative_transfer.parquet').num_rows, 1)
        summaries = pq.read_table(output / 'model_summary.parquet').to_pylist()
        self.assertTrue(all(row['n'] == 1 and row['ci95_low'] is None for row in summaries))
        self.assertEqual({row['summary_source'] for row in summaries}, {'validated_aggregate', 'derived_from_per_run'})
        with self.assertRaises(FileExistsError):
            export(self.results, output)

    def test_missing_families_have_typed_empty_tables_and_null_diagnostics(self):
        self.write('source_only_target_val/run.json', {**self.shared, 'target_development': self.metrics(.2), 'checkpoint': 'source.pt'})
        export(self.results, self.root / 'out')
        table = pq.read_table(self.root / 'out' / 'feature_shift.parquet')
        self.assertEqual(table.schema, SCHEMAS['feature_shift'])
        self.assertEqual(table.num_rows, 0)
        row = pq.read_table(self.root / 'out' / 'experiment_runs.parquet').to_pylist()[0]
        self.assertIsNone(row['gradient_cosine'])
        self.assertEqual(row['adaptation_gain'], 0)

    def test_stale_diagnostics_and_aggregate_are_rejected(self):
        path = self.fixtures()
        original = json.loads(path.read_text())
        path.write_text(json.dumps({**original, 'config_sha256': 'stale'}))
        with self.assertRaisesRegex(ValueError, 'Stale diagnostic'):
            build_tables(self.results)
        path.write_text(json.dumps(original))
        aggregate = self.results / 'aggregate/summary.json'
        data = json.loads(aggregate.read_text())
        data['summary'][0]['metrics']['pr_auc']['mean'] = .99
        aggregate.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, 'Stale aggregate summary'):
            build_tables(self.results)

    def test_noncanonical_tags_are_ignored_and_duplicates_rejected(self):
        self.fixtures()
        path = self.write('mmd/target_val_epoch0_proposal_mmd_v1/run.json', {'protocol': 'proposal_v1'})
        _, manifest = build_tables(self.results)
        self.assertIn(str(path), manifest['ignored_files'])
        existing = self.results / f'source_only_target_val/{self.direction}/seed42.json'
        self.write('source_only_target_val/duplicate.json', json.loads(existing.read_text()))
        with self.assertRaisesRegex(ValueError, 'Duplicate experiment'):
            build_tables(self.results)

    def test_dataset_manifest_avoids_direction_double_counting(self):
        self.fixtures()
        manifest = self.root / 'data_manifest.json'
        manifest.write_text(json.dumps({'protocol': 'proposal_v2', 'data_revision': 'spark_data_v1',
            'counts': {'class_counts': [{'domain': 'unsw', 'split': 'train', 'label': 0, 'count': 80},
                                       {'domain': 'unsw', 'split': 'train', 'label': 1, 'count': 20}]}}))
        tables, _ = build_tables(self.results, manifest)
        self.assertEqual(len(tables['dataset_summary']), 1)
        row = tables['dataset_summary'][0]
        self.assertIsNone(row['direction'])
        self.assertEqual(row['rows'], 100)
        self.assertEqual(row['data_revision'], 'spark_data_v1')

    @unittest.skipUnless(importlib.util.find_spec('duckdb'), 'Optional DuckDB extra is not installed')
    def test_duckdb_is_a_portable_snapshot(self):
        import duckdb
        self.fixtures()
        output = self.root / 'duck'
        export(self.results, output, duckdb=True)
        for parquet in output.glob('*.parquet'):
            parquet.unlink()
        with duckdb.connect(str(output / 'dashboard.duckdb'), read_only=True) as connection:
            self.assertEqual(connection.execute('SELECT count(*) FROM experiment_runs').fetchone()[0], 4)


if __name__ == '__main__':
    unittest.main()
