"""Export existing development JSON reports to Tableau CSVs; no model execution."""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
METHODS = {
    'proposal_mmd_v2': 'marginal_mmd',
    'proposal_mkmmd_v2': 'mk_mmd',
    'proposal_class_aware_v2': 'class_aware_mmd',
}


def flatten(value, prefix=''):
    """Flatten scalars; serialize residual lists without multiplying rows."""
    result = {}
    for key, item in value.items():
        name = f'{prefix}_{key}' if prefix else key
        if isinstance(item, dict):
            result.update(flatten(item, name))
        elif isinstance(item, list):
            result[name] = json.dumps(item, ensure_ascii=False)
        else:
            result[name] = item
    return result


def write_table(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    if not columns:
        print(f'No observations: {path.name}')
        return
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f'{path.name}: {len(rows)} rows')


def export(result_root, output):
    summary_path = result_root / 'aggregate/summary.json'
    aggregate = json.loads(summary_path.read_text(encoding='utf-8'))
    files = sorted((result_root / 'diagnostics_target_val').glob('*/*.json'))
    if not files:
        raise FileNotFoundError('No diagnostics_target_val/*/*.json reports found')
    output.mkdir(parents=True, exist_ok=True)
    write_table(output / 'performance_runs.csv', [
        {'data_role': 'target_validation', 'source_json': str(summary_path), **flatten(row)}
        for row in aggregate['per_run']
    ])
    summaries = []
    for row in aggregate['summary']:
        for metric, stats in row['metrics'].items():
            ci = stats.get('ci95')
            summaries.append({
                'direction': row['direction'], 'method': row['method'],
                'data_role': 'target_validation', 'metric': metric,
                'n_seeds': stats['n'], 'mean': stats['mean'], 'std': stats['std'],
                'ci95_low': ci[0] if ci else None,
                'ci95_high': ci[1] if ci else None,
                'ci_method': stats.get('ci_method'),
            })
    write_table(output / 'performance_summary.csv', summaries)

    alignment, metrics, quantiles, deltas, diagnostics = [], [], [], [], []
    curve_columns = ['direction', 'method', 'seed', 'data_role', 'source_json',
                     'phase', 'point_index', 'threshold', 'precision', 'recall', 'f1']
    curve_count = 0
    seen = set()
    missing_scores = []
    with (output / 'pr_curves.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        curve_writer = csv.DictWriter(stream, fieldnames=curve_columns)
        curve_writer.writeheader()
        for path in files:
            report = json.loads(path.read_text(encoding='utf-8'))
            config = Path(report['config']).stem
            if config not in METHODS:
                raise ValueError(f'Unknown diagnostic config: {path}: {config}')
            method = METHODS[config]
            identity = (report['direction'], method, report['seed'])
            if identity in seen:
                raise ValueError(f'Duplicate diagnostic run: {identity}')
            seen.add(identity)
            base = dict(zip(('direction', 'method', 'seed'), identity))
            base.update(data_role='target_validation', source_json=str(path))
            diagnostics.append({**base, **flatten({
                key: value for key, value in report.items()
                if key not in {'direction', 'seed', 'config', 'target_validation_score_analysis'}
            })})
            delta = dict(base)
            for phase in ('before', 'after'):
                mmd = report['marginal_alignment'][phase]
                auc = report['domain_separability'][phase]
                alignment.append({
                    **base, 'phase': phase, 'mmd2': mmd['mmd2_mean'],
                    'mmd2_subsample_std': mmd.get('mmd2_std'),
                    'domain_auc': auc['auc_mean'], 'domain_auc_fold_std': auc.get('auc_std'),
                })
            delta['delta_mmd2'] = (report['marginal_alignment']['after']['mmd2_mean']
                                   - report['marginal_alignment']['before']['mmd2_mean'])
            delta['delta_domain_auc'] = (report['domain_separability']['after']['auc_mean']
                                         - report['domain_separability']['before']['auc_mean'])
            scores = report.get('target_validation_score_analysis')
            if not scores:
                missing_scores.append(str(path))
            else:
                for phase in ('before', 'after'):
                    score = scores[phase]
                    metrics.append({**base, 'phase': phase, **flatten({
                        key: value for key, value in score.items()
                        if key not in {'score_distributions', 'pr_curve'}
                    })})
                    for label, distribution in score['score_distributions'].items():
                        for index, (quantile, value) in enumerate(distribution['quantiles'].items()):
                            quantiles.append({
                                **base, 'phase': phase, 'class': label,
                                'quantile': quantile, 'quantile_order': index,
                                'score': value, 'count': distribution['count'],
                                'fraction_below_source_threshold':
                                    distribution.get('fraction_below_source_threshold'),
                            })
                    curve = score['pr_curve']
                    lengths = {len(curve[key]) for key in ('threshold', 'precision', 'recall', 'f1')}
                    if len(lengths) != 1:
                        raise ValueError(f'PR curve lengths differ: {path}, {phase}')
                    for index, values in enumerate(zip(
                        curve['threshold'], curve['precision'], curve['recall'], curve['f1']
                    )):
                        curve_writer.writerow({**base, 'phase': phase, 'point_index': index,
                            **dict(zip(('threshold', 'precision', 'recall', 'f1'), values))})
                        curve_count += 1
                for name in ('average_precision', 'roc_auc'):
                    delta[f'delta_{name}'] = scores['after'][name] - scores['before'][name]
                for name in ('recall', 'fpr', 'f1'):
                    delta[f'delta_{name}'] = (
                        scores['after']['source_threshold_metrics'][name]
                        - scores['before']['source_threshold_metrics'][name])
            deltas.append(delta)
    for name, rows in (
        ('alignment_runs', alignment), ('score_metrics', metrics),
        ('score_quantiles', quantiles), ('diagnostic_deltas', deltas),
        ('diagnostics_runs', diagnostics),
    ):
        write_table(output / f'{name}.csv', rows)
    print(f'pr_curves.csv: {curve_count} rows')
    (output / 'export_manifest.json').write_text(json.dumps({
        'result_root': str(result_root), 'aggregate_json': str(summary_path),
        'diagnostic_files': [str(path) for path in files],
        'diagnostics_missing_score_analysis': missing_scores,
        'data_role': 'target_validation',
        'oracle_f1_role': 'diagnostic_only; not deployed threshold',
        'mmd_std_role': 'subsampling standard deviation, not 95% CI',
        'auc_std_role': 'CV fold standard deviation, not seed variation',
    }, indent=2), encoding='utf-8')
    if missing_scores:
        print(f'WARNING: {len(missing_scores)} reports lack score analysis; see export_manifest.json')
    print(f'Saved CSVs: {output}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result-root', type=Path, default=ROOT / 'results/canonical')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or args.result_root / 'tableau'
    if output.exists():
        raise FileExistsError(f'Export directory already exists: {output}; use a new --output')
    export(args.result_root, output)


if __name__ == '__main__':
    main()
