"""Export proposal_v2 research JSON as five typed BI tables and optional DuckDB."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import t
from training.data_revision import revision_path

ROOT = Path(__file__).resolve().parents[2]
METHODS = {'source_only', 'marginal_mmd', 'mk_mmd', 'class_aware_mmd'}
CONFIG_METHOD = {'proposal_mmd_v2': 'marginal_mmd', 'proposal_mkmmd_v2': 'mk_mmd',
                 'proposal_class_aware_v2': 'class_aware_mmd'}
FEATURES = ['flow_duration', 'fwd_packets', 'bwd_packets', 'fwd_bytes', 'bwd_bytes']
METRICS = ['pr_auc', 'roc_auc', 'macro_f1', 'recall', 'fpr']
S, I, D, B = pa.string(), pa.int64(), pa.float64(), pa.bool_()
SCHEMAS = {
    'experiment_runs': pa.schema([(key, S) for key in ('run_id', 'protocol', 'direction', 'method', 'phase')]
        + [('seed', I), ('lambda', D)] + [(key, D) for key in METRICS + ['adaptation_gain', 'mmd_before',
            'mmd_after', 'domain_auc_before', 'domain_auc_after', 'gradient_cosine']]
        + [(key, S) for key in ('config_sha256', 'preprocessor_sha256', 'checkpoint', 'source_file', 'diagnostic_file')]),
    'feature_shift': pa.schema([(key, S) for key in ('protocol', 'direction', 'phase', 'feature')]
        + [('seed', I)] + [(key, D) for key in ('ks_statistic', 'p_value', 'input_mmd', 'domain_auc', 'source_mean', 'target_mean', 'source_median', 'target_median')]
        + [('source_rows', I), ('target_rows', I), ('source_file', S)]),
    'negative_transfer': pa.schema([(key, S) for key in ('run_id', 'direction', 'method', 'phase')]
        + [('seed', I), ('lambda', D), ('detected', B)]
        + [(key, D) for key in ('adaptation_gain', 'mmd_before', 'mmd_after', 'domain_auc_before',
                               'domain_auc_after', 'gradient_cosine', 'gradient_negative_fraction', 'prior_gap')]
        + [('likely_causes_json', S), ('source_file', S)]),
    'dataset_summary': pa.schema([(key, S) for key in ('direction', 'domain', 'split', 'data_revision', 'split_sha256')]
        + [(key, I) for key in ('rows', 'normal_count', 'attack_count')]
        + [('count_source', S), ('source_file', S)]),
    'model_summary': pa.schema([(key, S) for key in ('direction', 'method', 'phase', 'config_sha256', 'metric')]
        + [('lambda', D), ('n', I)] + [(key, D) for key in ('mean', 'std', 'ci95_low', 'ci95_high')]
        + [(key, S) for key in ('ci_method', 'seeds_json', 'summary_source', 'source_file')]),
}


def read_json(path):
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f'Expected JSON object: {path}')
    return data


def number(value):
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('Nonfinite BI metric')
    return value


def nested(data, *keys):
    for key in keys:
        if not isinstance(data, dict):
            return None
        data = data.get(key)
    return data


def identity(data):
    if data.get('direction') not in ('unsw_to_cicids', 'cicids_to_unsw'):
        raise ValueError('Invalid direction')
    if not isinstance(data.get('seed'), int) or isinstance(data['seed'], bool):
        raise ValueError('Invalid seed')
    if data.get('method', 'source_only') not in METHODS:
        raise ValueError('Unsupported thesis method')
    if data.get('features', FEATURES) != FEATURES or data.get('feature_count', 5) != 5:
        raise ValueError('Expected five-feature proposal_v2 schema')
    protocol = data.get('protocol')
    if protocol is not None and protocol != 'proposal_v2':
        raise ValueError(f'Unsupported protocol: {protocol}')


def build_tables(result_root, dataset_manifest=None):
    """Read result metadata only. Missing artifact families yield typed empty tables."""
    result_root = Path(result_root)
    if not result_root.is_dir() or result_root.name == 'proposal_v1':
        raise ValueError('Expected an existing proposal_v2 result directory')
    tables = {name: [] for name in SCHEMAS}
    runs, raw_runs, summaries, diagnostics = {}, {}, [], []
    ignored, warnings, inputs = [], [], {}
    dataset_rows = {}

    def dataset(direction, domain, split, counts, path, fingerprint=None, revision=None):
        if counts is None:
            return
        normal, attack = map(int, counts)
        if normal < 0 or attack < 0:
            raise ValueError('Negative dataset counts')
        key = (direction, domain, split, fingerprint)
        row = dict(direction=direction, domain=domain, split=split, data_revision=revision,
                   split_sha256=fingerprint, rows=normal + attack, normal_count=normal,
                   attack_count=attack, count_source='full_split', source_file=str(path))
        prior = dataset_rows.get(key)
        if prior and (prior['normal_count'], prior['attack_count']) != (normal, attack):
            raise ValueError(f'Conflicting dataset counts: {key}')
        dataset_rows[key] = prior or row

    def add_run(data, path, phase, method, metrics, aggregate_fallback=False):
        data = {**data, 'method': method}
        identity(data)
        if phase not in ('development', 'final_test'):
            raise ValueError('Unsupported phase')
        if 'phase' in data and data['phase'] != phase:
            raise ValueError('Artifact phase does not match its metric family')
        direction, seed = data['direction'], data['seed']
        target = direction.split('_to_')[1]
        split_field = 'target_test_split' if phase == 'final_test' else 'target_development_split'
        expected_split = f'{target}_test' if phase == 'final_test' else f'{target}_val'
        if split_field in data and data[split_field] != expected_split:
            raise ValueError('Artifact uses the wrong target split')
        key = (direction, method, seed, phase)
        row = dict(protocol='proposal_v2', direction=direction, method=method, seed=seed,
                   phase=phase, **{metric: number(metrics[metric]) for metric in METRICS},
                   **{'lambda': 0.0 if method == 'source_only' else number(data.get('lambda_mmd'))},
                   config_sha256=data.get('config_sha256'), preprocessor_sha256=data.get('preprocessor_sha256'),
                   checkpoint=data.get('checkpoint'), source_file=str(path))
        if any(row[metric] is None or not 0 <= row[metric] <= 1 for metric in METRICS):
            raise ValueError('Expected finite metrics in [0, 1]')
        row['run_id'] = hashlib.sha256(json.dumps(key).encode()).hexdigest()[:20]
        if key in runs:
            if not aggregate_fallback:
                raise ValueError(f'Duplicate experiment: {key}')
            if any(not math.isclose(runs[key][m], row[m], abs_tol=1e-9) for m in METRICS):
                raise ValueError(f'Stale aggregate per_run metrics: {key}')
            return
        runs[key], raw_runs[key] = row, data
        source, target = direction.split('_to_')
        prepared = data.get('prepared_split_sha256', {})
        if phase == 'development' and not aggregate_fallback:
            dataset(direction, source, 'train', data.get('source_train_counts'), path, prepared.get('source_train'))
            for domain, split_name, values, fingerprint in (
                (source, 'val', data.get('source_val', data.get('within_domain')), prepared.get('source_val')),
                (target, 'val', metrics, prepared.get('target_val'))):
                if values and all(k in values for k in ('tn', 'fp', 'fn', 'tp')):
                    dataset(direction, domain, split_name, [values['tn'] + values['fp'], values['fn'] + values['tp']], path, fingerprint)
        elif phase == 'final_test' and all(k in metrics for k in ('tn', 'fp', 'fn', 'tp')):
            dataset(direction, target, 'test', [metrics['tn'] + metrics['fp'], metrics['fn'] + metrics['tp']], path, data.get('target_test_sha256'))

    for path in sorted(result_root.rglob('*.json')):
        parts = path.relative_to(result_root).parts
        family = parts[0]
        # Historical tags/config sweeps are outside the four-method frozen suite.
        if family == 'mmd' and (len(parts) < 2 or not any(parts[1] == f'target_val_epoch0_{stem}' for stem in CONFIG_METHOD)):
            ignored.append(str(path)); continue
        if family not in ('source_only_target_val', 'mmd', 'domain_shift', 'diagnostics_target_val', 'aggregate', 'final_target_test'):
            ignored.append(str(path)); continue
        data = read_json(path)
        inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        if family == 'source_only_target_val':
            add_run(data, path, 'development', 'source_only', data['target_development'])
        elif family == 'mmd':
            add_run(data, path, 'development', data['method'], data['cross_domain'])
        elif family == 'final_target_test':
            add_run(data, path, 'final_test', data['method'], data['target_test'])
        elif family == 'aggregate':
            if data.get('protocol') != 'proposal_v2' or data.get('data_role') != 'development_only':
                raise ValueError(f'Wrong aggregate protocol/phase: {path}')
            summaries.append((data, path))
        elif family == 'diagnostics_target_val':
            identity(data)
            diagnostics.append((data, path))
        else:
            identity(data)
            for feature, values in data['ks_by_feature'].items():
                if feature not in FEATURES:
                    raise ValueError(f'Unknown canonical feature: {feature}')
                tables['feature_shift'].append(dict(protocol='proposal_v2', direction=data['direction'],
                    seed=data['seed'], phase='pre_adaptation', feature=feature,
                    ks_statistic=number(values['ks_statistic']), p_value=number(values['p_value']),
                    **{key: number(values.get(key)) for key in ('source_mean', 'target_mean', 'source_median', 'target_median')},
                    input_mmd=number(nested(data, 'input_mmd', 'mmd2_mean')),
                    domain_auc=number(nested(data, 'domain_classifier', 'auc_mean')),
                    source_rows=nested(data, 'sampling', 'source_total_rows'),
                    target_rows=nested(data, 'sampling', 'target_total_rows'), source_file=str(path)))

    for data, path in summaries:
        for row in data.get('per_run', []):
            add_run(row, path, 'development', row['method'], row, aggregate_fallback=True)

    for key, row in runs.items():
        direction, method, seed, phase = key
        if phase == 'final_test':
            development = runs.get((direction, method, seed, 'development'))
            if development:
                if row['checkpoint'] != development['checkpoint']:
                    raise ValueError(f'Final/development checkpoint mismatch: {key}')
                for field in ('lambda', 'config_sha256', 'preprocessor_sha256'):
                    row[field] = development[field]
        baseline_key = (direction, 'source_only', seed, phase)
        baseline = runs.get(baseline_key)
        if baseline is None:
            warnings.append(f'No paired source-only {phase} baseline: {key}')
            continue
        original, source = raw_runs[key], raw_runs[baseline_key]
        if phase == 'development' and method != 'source_only':
            for field in ('preprocessor_sha256', 'common_feature_config_sha256', 'prepared_split_sha256'):
                if field in original and field in source and original[field] != source[field]:
                    raise ValueError(f'Unmatched baseline provenance: {key} {field}')
            if original.get('source_checkpoint') and baseline.get('checkpoint') and baseline['checkpoint'] != original['source_checkpoint']:
                raise ValueError(f'Unmatched source checkpoint: {key}')
        if phase == 'final_test' and original.get('target_test_sha256') != source.get('target_test_sha256'):
            raise ValueError(f'Unmatched final target-test split: {key}')
        row['adaptation_gain'] = row['pr_auc'] - baseline['pr_auc']

    for data, path in diagnostics:
        stem = Path(data.get('config', '')).stem
        method = CONFIG_METHOD.get(stem)
        key = (data['direction'], method, data['seed'], 'development')
        if key not in runs:
            warnings.append(f'Orphan/noncanonical diagnostic skipped: {path}'); continue
        row = runs[key]
        if row.get('diagnostic_file'):
            raise ValueError(f'Duplicate diagnostic: {key}')
        if not row.get('checkpoint') or data.get('config_sha256') != row.get('config_sha256') or nested(data, 'artifacts', 'adapted_checkpoint') != row['checkpoint']:
            raise ValueError(f'Stale diagnostic binding: {path}')
        observed_gain = number(nested(data, 'negative_transfer', 'delta_target_pr_auc'))
        if row.get('adaptation_gain') is not None and observed_gain is not None and not math.isclose(row['adaptation_gain'], observed_gain, abs_tol=1e-9):
            raise ValueError(f'Stale diagnostic adaptation gain: {path}')
        row.update(mmd_before=number(nested(data, 'marginal_alignment', 'before', 'mmd2_mean')),
                   mmd_after=number(nested(data, 'marginal_alignment', 'after', 'mmd2_mean')),
                   domain_auc_before=number(nested(data, 'domain_separability', 'before', 'auc_mean')),
                   domain_auc_after=number(nested(data, 'domain_separability', 'after', 'auc_mean')),
                   gradient_cosine=number(nested(data, 'gradient_conflict', 'mean')), diagnostic_file=str(path))
        tables['negative_transfer'].append({**row, 'detected': nested(data, 'negative_transfer', 'detected'),
            'gradient_negative_fraction': number(nested(data, 'gradient_conflict', 'negative_fraction')),
            'prior_gap': number(nested(data, 'label_prior', 'absolute_gap')),
            'likely_causes_json': json.dumps(data.get('likely_causes', [])), 'source_file': str(path)})

    if dataset_manifest:
        path = Path(dataset_manifest)
        data = read_json(path)
        if data.get('protocol') != 'proposal_v2':
            raise ValueError('Wrong dataset manifest protocol')
        inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        # The full data manifest is authoritative; do not duplicate its domain counts by direction.
        dataset_rows.clear()
        grouped = defaultdict(lambda: [0, 0])
        for row in data['counts']['class_counts']:
            if row['domain'] not in ('unsw', 'cicids') or row['split'] not in ('train', 'val', 'test') or row['label'] not in (0, 1):
                raise ValueError('Invalid manifest class count')
            grouped[(row['domain'], row['split'])][row['label']] += int(row['count'])
        for (domain, split), counts in grouped.items():
            dataset(None, domain, split, counts, path, revision=data.get('data_revision'))

    shift_keys = [(row['direction'], row['seed'], row['feature']) for row in tables['feature_shift']]
    if len(set(shift_keys)) != len(shift_keys):
        raise ValueError('Duplicate feature-shift observations')
    # Aggregate per_run omits method config metadata. Fill only from a unique
    # observed config for that same method/direction/phase; reject mixed suites.
    suite_groups = defaultdict(list)
    for row in runs.values():
        suite_groups[(row['direction'], row['method'], row['phase'])].append(row)
    for key, rows in suite_groups.items():
        for field in ('config_sha256', 'preprocessor_sha256', 'lambda'):
            observed = {row[field] for row in rows if row.get(field) is not None}
            if len(observed) > 1:
                raise ValueError(f'Mixed configuration/data for seed summary: {key} {field}')
            if observed:
                value = next(iter(observed))
                for row in rows:
                    if row.get(field) is None:
                        row[field] = value
    tables['experiment_runs'] = list(runs.values())
    tables['dataset_summary'] = list(dataset_rows.values())
    groups = defaultdict(list)
    for row in runs.values():
        groups[(row['direction'], row['method'], row['phase'], row['config_sha256'])].append(row)
    for (direction, method, phase, config_hash), rows in sorted(groups.items(), key=lambda item: str(item[0])):
        rows.sort(key=lambda row: row['seed'])
        for metric in METRICS + ['adaptation_gain']:
            valid = [row for row in rows if row.get(metric) is not None]
            if not valid:
                continue
            values = np.array([row[metric] for row in valid])
            n, mean = len(values), float(values.mean())
            std = float(values.std(ddof=1)) if n > 1 else 0.0
            width = float(t.ppf(.975, n-1) * std / math.sqrt(n)) if n > 1 else None
            tables['model_summary'].append(dict(direction=direction, method=method, phase=phase,
                config_sha256=config_hash, metric=metric, **{'lambda': rows[0]['lambda']}, n=n, mean=mean, std=std,
                ci95_low=mean-width if width is not None else None, ci95_high=mean+width if width is not None else None,
                ci_method='student_t_across_paired_seeds' if metric == 'adaptation_gain' else 'student_t_across_seeds',
                seeds_json=json.dumps([row['seed'] for row in valid]), summary_source='derived_from_per_run', source_file=None))
    # Aggregate summaries must agree with the exported development facts, not count them twice.
    for data, path in summaries:
        for summary in data['summary']:
            for metric, stats in summary['metrics'].items():
                column = 'adaptation_gain' if metric == 'adaptation_gain_pr_auc' else metric
                matches = [row for row in tables['model_summary'] if row['direction'] == summary['direction'] and row['method'] == summary['method'] and row['phase'] == 'development' and row['metric'] == column]
                if len(matches) != 1:
                    raise ValueError(f'Ambiguous/missing aggregate summary: {path}')
                row = matches[0]
                if json.loads(row['seeds_json']) != sorted(summary['seeds']) or any(not math.isclose(row[field], stats[field], abs_tol=1e-8) for field in ('mean', 'std')):
                    raise ValueError(f'Stale aggregate summary: {path}')
                row['summary_source'], row['source_file'] = 'validated_aggregate', str(path)
    return tables, {'protocol': 'proposal_v2', 'inputs_sha256': inputs, 'ignored_files': ignored,
                    'warnings': warnings, 'row_counts': {name: len(rows) for name, rows in tables.items()},
                    'small_seed_caveat': '95% t intervals reflect run-seed variation on fixed datasets; small n gives uncertain intervals.'}


def export(result_root=revision_path('result_root', ROOT / 'results/proposal_v2'), output=ROOT / 'analytics', dataset_manifest=None, duckdb=False):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Use a fresh BI snapshot directory: {output}')
    if duckdb:
        try:
            import duckdb as db
        except ImportError as error:
            raise RuntimeError("Install the analytics extra: pip install -e '.[analytics]'") from error
    tables, manifest = build_tables(result_root, dataset_manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.dashboard-', dir=output.parent))
    try:
        statements = []
        for name, schema in SCHEMAS.items():
            table = pa.Table.from_pylist(tables[name], schema=schema)
            pq.write_table(table, staging / f'{name}.parquet', compression='snappy')
            parquet_path = str(output / f'{name}.parquet').replace("'", "''")
            statements.append(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet('{parquet_path}');")
        (staging / 'create_views.sql').write_text('\n'.join(statements)+'\n')
        if duckdb:
            with db.connect(str(staging / 'dashboard.duckdb')) as connection:
                for name in SCHEMAS:
                    connection.execute(f'CREATE TABLE {name} AS SELECT * FROM read_parquet(?)', [str(staging / f'{name}.parquet')])
        (staging / '_manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')
        if output.exists():
            raise FileExistsError(f'BI snapshot appeared during export: {output}')
        staging.rename(output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', type=Path, default=revision_path('result_root', ROOT / 'results/proposal_v2'))
    parser.add_argument('--output', type=Path, default=ROOT / 'analytics')
    parser.add_argument('--dataset-manifest', type=Path)
    parser.add_argument('--duckdb', action='store_true', help='Also materialize portable dashboard.duckdb tables')
    args = parser.parse_args()
    print(json.dumps(export(args.results_root, args.output, args.dataset_manifest, args.duckdb), indent=2))


if __name__ == '__main__':
    main()
