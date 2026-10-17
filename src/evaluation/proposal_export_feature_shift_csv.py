"""Export intrinsic sampling diagnostics to Tableau KS and exact ECDF CSVs."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp
from project_utils import sha256

ROOT = Path(__file__).resolve().parents[2]


def export(result_root, output, seeds):
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Sampling seeds must be nonempty and unique')
    reports = []
    features = None
    revision = None
    for seed in seeds:
        folder = result_root / 'domain_shift/intrinsic' / f'seed{seed}'
        report = json.loads((folder / 'result.json').read_text())
        identity = (report['data_revision'], report['data_manifest_sha256'])
        if report['seed'] != seed or report['intrusion_labels_used']:
            raise ValueError(f'Invalid sampling protocol: seed{seed}')
        if features is not None and (report['features'] != features or identity != revision):
            raise ValueError('Mixed feature schemas or data revisions')
        features, revision = report['features'], identity
        sample = folder / 'samples.npz'
        if sha256(sample) != report['sample_artifact_sha256']:
            raise ValueError(f'Sample artifact hash mismatch: seed{seed}')
        reports.append((seed, folder, report))
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    ecdf_count = 0
    with (output / 'ecdf_common_features.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            'feature', 'domain', 'sampling_seed', 'sample_n', 'point_order',
            'value_raw', 'value_log1p', 'ecdf',
        ])
        writer.writeheader()
        for seed, folder, report in reports:
            with np.load(folder / 'samples.npz') as samples:
                seed_rows = []
                for index, feature in enumerate(features):
                    values = {}
                    for domain in ('unsw', 'cicids'):
                        values[domain] = samples[f'{domain}_raw'][:, index].copy()
                        values[domain][~np.isfinite(values[domain])] = report['shared_transformation']['medians'][index]
                        if not np.isfinite(values[domain]).all():
                            raise ValueError(f'Nonfinite imputed values: {seed}/{feature}')
                        if len(values[domain]) != report['sample_rows'][domain]:
                            raise ValueError('Sample row count mismatch')
                    ks = float(ks_2samp(values['unsw'], values['cicids']).statistic)
                    if not np.isclose(ks, report['ks_by_feature'][feature]['ks_statistic'], atol=1e-12, rtol=0):
                        raise ValueError(f'KS mismatch: {seed}/{feature}')
                    seed_rows.append(dict(feature=feature, ks_statistic=ks, sampling_seed=seed,
                                          unsw_n=len(values['unsw']), cicids_n=len(values['cicids'])))
                    high = max(value.max() for value in values.values())
                    for domain, value in values.items():
                        unique, counts = np.unique(value, return_counts=True)
                        cumulative = np.cumsum(counts) / len(value)
                        previous = 0.0
                        point = 0
                        base = dict(feature=feature, domain='UNSW-NB15' if domain == 'unsw' else 'CICIDS2017',
                                    sampling_seed=seed, sample_n=len(value))
                        for x, probability in zip(unique, cumulative):
                            for y in (previous, float(probability)):
                                writer.writerow(dict(base, point_order=point, value_raw=float(x),
                                                     value_log1p=float(np.sign(x) * np.log1p(abs(x))), ecdf=y))
                                point += 1
                            previous = float(probability)
                        writer.writerow(dict(base, point_order=point, value_raw=float(high),
                                             value_log1p=float(np.sign(high) * np.log1p(abs(high))), ecdf=1.0))
                        ecdf_count += point + 1
                seed_rows.sort(key=lambda row: row['ks_statistic'], reverse=True)
                rows.extend(dict(rank=rank, **row) for rank, row in enumerate(seed_rows, 1))
    summaries = []
    for feature in features:
        values = np.array([row['ks_statistic'] for row in rows if row['feature'] == feature])
        summaries.append(dict(feature=feature, n_sampling_seeds=len(values), ks_mean=float(values.mean()),
                              ks_std=float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                              ks_min=float(values.min()), ks_max=float(values.max())))
    summaries.sort(key=lambda row: row['ks_mean'], reverse=True)
    summaries = [dict(rank=rank, **row) for rank, row in enumerate(summaries, 1)]
    for name, data in [('ks_ranking_common_features', rows), ('ks_summary_common_features', summaries)]:
        with (output / f'{name}.csv').open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    manifest = dict(sampling_seeds=seeds, features=features, data_revision=revision[0],
                    data_manifest_sha256=revision[1], ks_rows=len(rows), ecdf_rows=ecdf_count,
                    sources=[str((folder / 'result.json').resolve()) for _, folder, _ in reports],
                    ks_verified_against_json=True,
                    ecdf='Exact step ECDF with duplicate x coordinates; order by point_order separately for each feature/domain/sampling_seed.',
                    transform='Shared median imputation from each intrinsic report; value_log1p is signed_log1p of raw common features.',
                    units='flow_duration: seconds; packets: counts; bytes: bytes',
                    uncertainty='KS std describes overlapping random subsamples of fixed training datasets, not model training variation.')
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(output=str(output.resolve()), **manifest), indent=2))
    print(json.dumps(summaries, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result-root', type=Path, default=ROOT / 'results/canonical')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/analysis/tableau_feature_shift_5seeds')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46])
    args = parser.parse_args()
    export(args.result_root, args.output, args.seeds)


if __name__ == '__main__':
    main()
