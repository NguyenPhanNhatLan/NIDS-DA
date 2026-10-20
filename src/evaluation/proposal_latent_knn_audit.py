"""Balanced validation kNN diagnostics in original latent space; no training."""
import argparse
import csv
import json
from pathlib import Path

from project_utils.result_files import publish_results, result_file

import numpy as np
import pyarrow.parquet as pq
from sklearn.neighbors import NearestNeighbors

from evaluation.proposal_aggregate import result_path
from project_utils.statistics import seed_statistics
from evaluation.proposal_negative_transfer import (
    FEATURE_ROOT, ROOT, collect_representation, domains, load_models, sha256,
)
from features.parquet_vectors import vector_matrix

CONFIGS = {'marginal_mmd': 'proposal_mmd_v2', 'mk_mmd': 'proposal_mkmmd_v2',
           'class_aware_mmd': 'proposal_class_aware_v2'}


def balanced_sample(folder, n, seed):
    xs, ys = [], []
    files = sorted(folder.glob('*.parquet'))
    for path in files:
        for batch in pq.ParquetFile(path).iter_batches(columns=['features', 'label']):
            xs.append(vector_matrix(batch.column('features'), path))
            ys.append(batch.column('label').to_numpy(zero_copy_only=False))
    x, y = np.concatenate(xs), np.concatenate(ys)
    if not np.isfinite(x).all() or not np.isin(y, [0, 1]).all():
        raise ValueError('Invalid prepared validation data')
    rng = np.random.default_rng(seed)
    selected = []
    for label in [0, 1]:
        pool = np.flatnonzero(y == label)
        if len(pool) < n:
            raise ValueError(f'{folder}: need {n} distinct flows of class {label}')
        selected.extend(rng.choice(pool, n, replace=False).tolist())
    selected = np.sort(selected)
    return x[selected], y[selected], selected, [dict(path=str(p), sha256=sha256(p)) for p in files]


def agreements(query, reference, query_y, reference_y, ks, metric, same_domain=False):
    extra = 1 if same_domain else 0
    nearest = NearestNeighbors(n_neighbors=max(ks) + extra, metric=metric,
                               algorithm='brute', n_jobs=1).fit(reference)
    indices = nearest.kneighbors(query, return_distance=False)
    if same_domain:

        indices = np.array([row[row != i][:max(ks)] for i, row in enumerate(indices)])
    return {k: (reference_y[indices[:, :k]] == query_y[:, None]).mean(axis=1) for k in ks}


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--direction', choices=['both', 'unsw_to_cicids', 'cicids_to_unsw'], default='both')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46])
    parser.add_argument('--sampling-seed', type=int, default=42)
    parser.add_argument('--per-class', type=int, default=1000)
    parser.add_argument('--k', type=int, nargs='+', default=[15, 30])
    parser.add_argument('--metric', choices=['euclidean', 'cosine'], default='euclidean')
    parser.add_argument('--umap-seeds', type=int, nargs='*', default=[])
    parser.add_argument('--output', type=Path, default=ROOT / 'results/analysis/latent_knn_balanced')
    args = parser.parse_args()
    if not args.seeds or len(set(args.seeds)) != len(args.seeds) or args.per_class < 2:
        parser.error('Seeds must be unique/nonempty; per-class must be >= 2')
    if not args.k or min(args.k) < 1 or max(args.k) >= 2 * args.per_class:
        parser.error('Require 1 <= k < sampled flows per domain')
    if not set(args.umap_seeds) <= set(args.seeds):
        parser.error('umap-seeds must be included in seeds')
    if result_file(args.output / "manifest.json").exists():
        raise FileExistsError("kNN outputs already exist; choose another --output name")
    args.output.mkdir(parents=True, exist_ok=False)
    directions = ['unsw_to_cicids', 'cicids_to_unsw'] if args.direction == 'both' else [args.direction]
    rows, points, embeddings, provenance = [], [], [], []
    for direction in directions:
        source, target = domains(direction)
        xs, ys, ids_s, files_s = balanced_sample(FEATURE_ROOT / direction / f'{source}_val', args.per_class, args.sampling_seed)
        xt, yt, ids_t, files_t = balanced_sample(FEATURE_ROOT / direction / f'{target}_val', args.per_class, args.sampling_seed + 1)
        np.savez_compressed(args.output / f'{direction}_fixed_sample.npz', source_x=xs, source_y=ys,
                            source_row_indices=ids_s, target_x=xt, target_y=yt, target_row_indices=ids_t)
        provenance.append(dict(direction=direction, source_files=files_s, target_files=files_t))
        for seed in args.seeds:
            models = {}
            for method, tag in CONFIGS.items():
                baseline, adapted, _, _, paths = load_models(direction, seed, ROOT / 'configs' / f'{tag}.json')
                models.setdefault('source_only', baseline)
                models[method] = adapted
                provenance.append(dict(direction=direction, seed=seed, method=method, **paths))
            for method, model in models.items():
                print(f'kNN: {direction} seed{seed} {method}', flush=True)
                zs, _ = collect_representation(model, xs)
                zt, _ = collect_representation(model, xt)
                full = json.loads(result_path(method, direction, seed).read_text())
                ap = full['target_development' if method == 'source_only' else 'cross_domain']['pr_auc']
                tasks = [
                    ('cross_domain', 'Target', 'Source', zt, zs, yt, ys, ids_t, False),
                    ('within_domain', 'Source', 'Source', zs, zs, ys, ys, ids_s, True),
                    ('within_domain', 'Target', 'Target', zt, zt, yt, yt, ids_t, True),
                ]
                for kind, query_role, reference_role, zq, zr, yq, yr, ids, same in tasks:
                    result = agreements(zq, zr, yq, yr, args.k, args.metric, same)
                    for k, values in result.items():
                        base = dict(direction=direction, method=method, training_seed=seed, k=k,
                                    comparison=kind, query_role=query_role, reference_role=reference_role)
                        for label, name in [(0, 'Benign'), (1, 'Attack'), (None, 'Macro')]:
                            select = np.ones(len(yq), dtype=bool) if label is None else yq == label
                            agreement = float(values[select].mean())
                            rows.append(dict(base, class_group=name, agreement=agreement,
                                             mixing=1 - agreement, n_query=int(select.sum()), target_ap=ap))
                        for i, value in enumerate(values):
                            points.append(dict(base, flow_id=f'{query_role}_val_row{ids[i]}',
                                               class_group='Benign' if yq[i] == 0 else 'Attack',
                                               agreement=float(value), mixing=1 - float(value)))
                if seed in args.umap_seeds:
                    import umap
                    latent = np.concatenate([zs, zt])
                    coordinates = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=42,
                                             n_jobs=1).fit_transform(latent)
                    if not np.isfinite(coordinates).all():
                        raise ValueError('Invalid UMAP coordinates')
                    for i, xy in enumerate(coordinates):
                        is_source = i < len(ys)
                        local = i if is_source else i - len(ys)
                        label = (ys if is_source else yt)[local]
                        row_id = (ids_s if is_source else ids_t)[local]
                        role = 'Source' if is_source else 'Target'
                        embeddings.append(dict(direction=direction, method=method, training_seed=seed,
                                               flow_id=f'{role}_val_row{row_id}', role=role,
                                               class_group='Benign' if label == 0 else 'Attack',
                                               umap_1=float(xy[0]), umap_2=float(xy[1])))
    baseline = {(r['direction'], r['training_seed'], r['k'], r['comparison'], r['query_role'], r['class_group']): r
                for r in rows if r['method'] == 'source_only'}
    for row in rows:
        b = baseline[(row['direction'], row['training_seed'], row['k'], row['comparison'], row['query_role'], row['class_group'])]
        row['delta_agreement'] = row['agreement'] - b['agreement']
        row['delta_mixing'] = row['mixing'] - b['mixing']
        row['delta_ap'] = row['target_ap'] - b['target_ap']
    write_csv(args.output / 'knn_runs.csv', rows)
    write_csv(args.output / 'knn_flow_scores.csv', points)
    groups = {}
    fields = ['direction', 'method', 'k', 'comparison', 'query_role', 'reference_role', 'class_group']
    for row in rows:
        groups.setdefault(tuple(row[f] for f in fields), []).append(row)
    summaries = []
    for key, group in groups.items():
        for metric in ['agreement', 'mixing', 'delta_agreement', 'delta_mixing', 'delta_ap']:
            stats = seed_statistics([r[metric] for r in group])
            ci = stats['ci95']
            summaries.append(dict(zip(fields, key), metric=metric, n_seeds=stats['n'],
                                  mean=stats['mean'], std=stats['std'],
                                  ci95_low=ci[0] if ci else None, ci95_high=ci[1] if ci else None))
    write_csv(args.output / 'knn_summary.csv', summaries)
    if embeddings:
        write_csv(args.output / 'balanced_umap.csv', embeddings)
    manifest = dict(training_seeds=args.seeds, sampling_seed=args.sampling_seed, per_domain_class=args.per_class,
                    k=args.k, metric=args.metric, labels_used_training=False, labels_used_umap_fit=False,
                    data_role='post_hoc_validation_diagnostic_only', same_fixed_flows_across_models_and_seeds=True,
                    umap_seeds=args.umap_seeds, provenance=provenance,
                    interpretation='Agreement is mean fraction of reference neighbors sharing query class; mixing=1-agreement. Macro uses balanced classes. Chance label agreement is 0.5 for balanced references.',
                    limitations='Duplicate feature vectors are retained; self row excluded within domain, other duplicates remain. Tied neighbors use sklearn ordering. Seed CI covers model run variation conditional on fixed sample, not dataset uncertainty. Independent UMAP coordinates are not comparable across panels. Raw latent vectors are not rescaled.')
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    publish_results(args.output)
    print('Saved kNN outputs by file type under results/.', flush=True)


if __name__ == '__main__':
    main()
