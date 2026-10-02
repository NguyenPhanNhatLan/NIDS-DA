"""Train-only semantic feature-pair ranking; never projects target labels."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.feature_selection import mutual_info_classif

ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN = {'label', 'binary_label', 'label_clean', 'attack_cat', 'id', 'features'}
V1_WEIGHTS = {'KS_similarity': .40, 'W_similarity': .35,
              'MI_similarity': .15, 'quality_score': .10}


def validate(config, pairs):
    if config['weights'] != V1_WEIGHTS or config['lambda_redundancy'] != .20:
        raise ValueError('V1 weights and redundancy penalty are fixed.')
    if config['source_domain'] not in ('unsw', 'cicids'):
        raise ValueError('Unknown source domain')
    if config['sample_size'] < 4 or not config['seeds']:
        raise ValueError('Need samples >= 4 and at least one seed')
    if len({p['canonical_name'] for p in pairs}) != len(pairs):
        raise ValueError('Duplicate canonical names')
    for p in pairs:
        if type(p['semantic_valid']) is not bool:
            raise ValueError('semantic_valid must be boolean')
        if any(p[d].lower() in FORBIDDEN for d in ('unsw', 'cicids')):
            raise ValueError('Labels, IDs and standardized vectors cannot be candidates')


def convert(values, pair, domain):
    """Convert physical units without fitting or scaling a domain distribution."""
    x = pd.to_numeric(pd.Series(values), errors='coerce').to_numpy(dtype=float, copy=True)
    conversion = pair['conversion']
    if conversion == 'cicids_us_to_seconds':
        x = x / (1e6 if domain == 'cicids' else 1)
    elif conversion == 'iat_to_seconds':
        x = x / (1e6 if domain == 'cicids' else 1e3)
    elif conversion != 'identity':
        raise ValueError(f'Unknown conversion: {conversion}')
    return x


def clean(values, pair, domain, config):
    x = convert(values, pair, domain)
    missing = np.isnan(x)
    invalid = np.isinf(x) | (x < 0)
    finite = ~(missing | invalid)
    observed = x[finite]
    constant = len(np.unique(observed)) <= 1
    dominant = (np.unique(observed, return_counts=True)[1].max() / len(observed)
                if len(observed) else 1.)
    z = np.log1p(observed) if pair.get('log1p', False) else observed
    near = constant or dominant >= config['near_constant_fraction'] or (
        len(z) > 0 and np.var(z) <= config['variance_threshold'])
    quality = float(finite.mean()) * (0. if constant else .1 if near else 1.)
    report = dict(missing_ratio=float(missing.mean()), invalid_ratio=float(invalid.mean()),
                  finite_value_ratio=float(finite.mean()), constant=bool(constant),
                  near_zero_variance=bool(near), dominant_fraction=float(dominant),
                  quality_score=quality)
    x[~finite] = np.nan
    return x, report


def transform(x, pair, median):
    x = np.where(np.isfinite(x), x, median)
    return np.log1p(x) if pair.get('log1p', False) else x


def distribution_similarity(source, target, epsilon=1e-9):
    ks = 1. - float(ks_2samp(source, target, method='asymp').statistic)
    pooled = np.concatenate([source, target])
    iqr = float(np.diff(np.quantile(pooled, [.25, .75]))[0])
    distance = float(wasserstein_distance(source, target)) / (iqr + epsilon)
    return ks, float(np.exp(-distance)), distance


def greedy_select(scores, correlation, penalty=.20):
    remaining = {r['canonical_name']: dict(r) for r in scores}
    chosen = []
    while remaining:
        options = []
        for name, row in remaining.items():
            redundancy = max((float(correlation.loc[name, r['canonical_name']])
                              for r in chosen), default=0.)
            options.append(dict(row, redundancy=redundancy,
                                selection_gain=row['equivalence_score'] - penalty * redundancy))
        best = min(options, key=lambda r: (-r['selection_gain'], r['score_std'], r['canonical_name']))
        best['rank'] = len(chosen) + 1
        chosen.append(best)
        del remaining[best['canonical_name']]
    return chosen


def score_pairs(unsw, cicids, pairs, config):
    """DataFrame API: only the source's label is accessed, in either direction."""
    validate(config, pairs)
    frames = {'unsw': unsw, 'cicids': cicids}
    source = config['source_domain']
    target = 'cicids' if source == 'unsw' else 'unsw'
    y = frames[source]['label'].to_numpy()
    if not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
        raise ValueError('Source training labels must contain both binary classes')
    if min(len(unsw), len(cicids)) < 4:
        raise ValueError('Need at least four training rows per domain')
    arrays = {d: {} for d in frames}
    records, eligible = [], []
    for p in pairs:
        row = dict(canonical_name=p['canonical_name'], unsw_feature=p['unsw'],
                   cicids_feature=p['cicids'], semantic_valid=p['semantic_valid'],
                   semantic_reason=p['reason'], unit_conversion=p['conversion'])
        records.append(row)
        if not p['semantic_valid']:
            row['status'] = 'semantic_rejected'
            continue
        absent = [f'{d}.{p[d]}' for d in frames if p[d] not in frames[d]]
        if absent:
            row['status'] = 'missing_columns: ' + ', '.join(absent)
            continue
        raw, quality = {}, {}
        for d in frames:
            raw[d], quality[d] = clean(frames[d][p[d]], p, d, config)
            row.update({f'{d}_{k}': v for k, v in quality[d].items()})
        if any(q['constant'] for q in quality.values()):
            row['status'] = 'constant_or_empty'
            continue
        # One source-fit median, in physical units, used for BOTH domains.
        median = float(np.nanmedian(raw[source]))
        row['imputation_median'] = median
        row['quality_score'] = min(q['quality_score'] for q in quality.values())
        for d in frames:
            arrays[d][p['canonical_name']] = transform(raw[d], p, median)
        row['status'] = 'eligible'
        eligible.append((p, row))
    if not eligible:
        return pd.DataFrame(records), [], []
    names = [p['canonical_name'] for p, _ in eligible]
    matrices = {d: pd.DataFrame(arrays[d])[names] for d in frames}
    mi_indices = np.random.default_rng(config['mi_seed']).choice(
        len(y), min(len(y), config['sample_size']), replace=False)
    if len(np.unique(y[mi_indices])) != 2:
        raise ValueError('MI sample contains only one source class; increase sample_size')
    mi_input = matrices[source].iloc[mi_indices].copy()
    for p, _ in eligible:
        if p.get('discrete', False):
            name = p['canonical_name']
            mi_input[name] = pd.factorize(mi_input[name], sort=True)[0]
    mi = mutual_info_classif(mi_input, y[mi_indices],
                            discrete_features=[p.get('discrete', False) for p, _ in eligible],
                            random_state=config['mi_seed'])
    mi_norm = mi / mi.max() if mi.max() > 0 else np.zeros_like(mi)
    trials = {n: [] for n in names}
    correlations, seed_reports = [], []
    for seed in config['seeds']:
        samples = {}
        for d in frames:
            idx = np.random.default_rng(seed).choice(len(frames[d]),
                min(len(frames[d]), config['sample_size']), replace=False)
            samples[d] = matrices[d].iloc[idx]
        corr = (samples[source].corr(method='spearman').abs() +
                samples[target].corr(method='spearman').abs()) / 2
        # Undefined correlation is conservative maximal redundancy.
        correlations.append(corr.fillna(1.))
        for i, (p, row) in enumerate(eligible):
            n = p['canonical_name']
            ks, ws, wn = distribution_similarity(samples[source][n], samples[target][n], config['epsilon'])
            metrics = dict(KS_similarity=ks, W_similarity=ws,
                           MI_similarity=float(mi_norm[i]), quality_score=row['quality_score'])
            score = sum(config['weights'][k] * v for k, v in metrics.items())
            trials[n].append((score, ks, ws, wn))
            seed_reports.append(dict(canonical_name=n, seed=seed, score=score,
                                     W_normalized=wn, **metrics))
    for i, (p, row) in enumerate(eligible):
        values = np.asarray(trials[p['canonical_name']])
        means, stds = values.mean(axis=0), values.std(axis=0)
        row.update(score_mean=float(means[0]), score_std=float(stds[0]),
                   ks_mean=float(means[1]), ks_std=float(stds[1]),
                   wasserstein_mean=float(means[2]), wasserstein_std=float(stds[2]),
                   W_normalized_mean=float(means[3]), MI=float(mi[i]), MI_similarity=float(mi_norm[i]),
                   KS_similarity=float(means[1]), W_similarity=float(means[2]),
                   equivalence_score=float(means[0]))
    selected = greedy_select([r for _, r in eligible], sum(correlations) / len(correlations),
                             config['lambda_redundancy'])
    by_name = {r['canonical_name']: r for r in selected}
    records = [by_name.get(r['canonical_name'], r) for r in records]
    return pd.DataFrame(records), selected, seed_reports


def load_training(config, pairs, root=ROOT):
    """Project columns at the parquet reader, so target labels never enter memory."""
    validate(config, pairs)
    frames, manifest = {}, {}
    for d in ('unsw', 'cicids'):
        path = (root / config['training_paths'][d]).resolve()
        if path.name != f'{d}_train' or path.parent.name != 'splits':
            raise ValueError('Only named raw training splits are accepted (data/splits/*_train)')
        files = sorted(path.glob('*.parquet'))
        if not files:
            raise FileNotFoundError(f'No parquet files: {path}')
        schema = pq.read_schema(files[0]).names
        cols = sorted({p[d] for p in pairs if p['semantic_valid'] and p[d] in schema})
        if d == config['source_domain']:
            cols.append('label')
        frames[d] = pd.concat([pq.read_table(f, columns=cols).to_pandas() for f in files], ignore_index=True)
        if config.get('deterministic_row_order', False):
            # Opt-in for regenerated Spark datasets: physical partition order can
            # change. Only projected features (plus source labels) participate.
            frames[d] = frames[d].sort_values(cols, kind='mergesort').reset_index(drop=True)
        # Hash only loaded columns: manifest also remains target-label independent.
        digest = hashlib.sha256(pd.util.hash_pandas_object(frames[d], index=False).values.tobytes()).hexdigest()
        manifest[d] = dict(path=str(path), rows=len(frames[d]), columns=cols,
                           projected_data_sha256=digest, files=[f.name for f in files])
    return frames, manifest


def run(config_path):
    config_path = Path(config_path).resolve()
    root = config_path.parent.parent
    config = json.loads(config_path.read_text())
    candidate_doc = json.loads((root / config['candidates']).read_text())
    pairs = candidate_doc['pairs']
    frames, inputs = load_training(config, pairs, root)
    scores, selected, trials = score_pairs(frames['unsw'], frames['cicids'], pairs, config)
    out = root / config['output_dir']
    cfg_out = root / config['config_output_dir']
    out.mkdir(parents=True, exist_ok=True)
    cfg_out.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out / 'feature_pair_scores.csv', index=False)
    pd.DataFrame(trials).to_csv(out / 'seed_scores.csv', index=False)
    subsets = {}
    for k in config['top_k']:
        rows = selected[:k]
        status = 'complete' if len(rows) == k else 'insufficient_eligible_pairs'
        subsets[str(k)] = dict(requested_k=k, actual_k=len(rows), status=status)
        pd.DataFrame(rows, columns=list(selected[0]) if selected else ['canonical_name']).to_csv(out / f'top{k}.csv', index=False)
        chosen_names = {r['canonical_name'] for r in rows}
        artifact = dict(version=1, source_domain=config['source_domain'], **subsets[str(k)],
                        features=rows, pairs=[p for p in pairs if p['canonical_name'] in chosen_names])
        (cfg_out / f'common_features_top{k}.json').write_text(json.dumps(artifact, indent=2, allow_nan=False)+'\n')
    manifest = dict(protocol=config, candidates=candidate_doc, inputs=inputs, subsets=subsets,
                    versions={p: importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','pyarrow']},
                    historical_limitation='CICIDS was correlation-pruned before splitting; no claim of end-to-end train-only preprocessing.')
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(f'Semantic candidates : {len(pairs)}\nValid candidates    : {sum(p["semantic_valid"] for p in pairs)}\nRejected candidates : {sum(not p["semantic_valid"] for p in pairs)}\nAvailable eligible  : {len(selected)}')
    if selected:
        print(pd.DataFrame(selected)[['rank','unsw_feature','cicids_feature','equivalence_score','score_std']].to_string(index=False))
    for k, summary in subsets.items():
        print(f'Top{k}: {summary["actual_k"]}/{k} — {summary["status"]}')
    return scores, selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT / 'configs/feature_selection_v1.json'))
    run(parser.parse_args().config)


if __name__ == '__main__':
    main()
