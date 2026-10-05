"""Post-hoc labeled DEVELOPMENT analysis of V2 -> V5f ranking changes. No fitting."""
import argparse
import json
import math

import numpy as np
import pyarrow.parquet as pq
import torch
from scipy.stats import rankdata

from evaluation.hda_v5b_calibration import data_snapshot, payload_hash
from evaluation.hda_v5f import calibration_path, dependencies, load_v5b_calibration, validate_affine, verify_training_data
from training.hda_v5f import load_context, load_student, load_v5d_reference
from training.thesis_protocol import evaluation_target, resolve_path
from training.v6_data import make_loader


def ap_contributions(scores, labels):
    """Per-positive contribution precision(score threshold)/total positives; ties grouped.

    Contributions sum to sklearn's non-interpolated average precision.
    This is a descriptive additive allocation, not a causal AP attribution.
    """
    positives = int(labels.sum())
    if positives == 0:
        return None
    order = np.argsort(-scores, kind='stable')
    sorted_scores, sorted_labels = scores[order], labels[order]
    ends = np.r_[np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:]), len(scores) - 1]
    starts = np.r_[0, ends[:-1] + 1]
    precision = np.cumsum(sorted_labels)[ends] / (ends + 1)
    allocation = np.repeat(precision, ends - starts + 1) * sorted_labels / positives
    result = np.empty(len(scores), dtype=np.float64)
    result[order] = allocation
    return result


def group_masks(delta):
    return {'large down': delta < -25,
            'down': (delta >= -25) & (delta < -10),
            'stable': np.abs(delta) <= 10,
            'up': (delta > 10) & (delta <= 25),
            'large up': delta > 25}


def mean_median(values):
    return {'mean': float(np.mean(values)), 'median': float(np.median(values))}


def origin_bins(x, y, labels, delta):
    """Bin by V2 scores on ALL development rows, never re-quantile a subgroup."""
    levels = [.02, .25, .50, .75, .95, .98]
    names = ['0-2%', '2-25%', '25-50%', '50-75%', '75-95%', '95-98%', '98-100%']
    cuts = np.quantile(x, levels)
    bins = np.searchsorted(cuts, x, side='right')
    bins[x <= cuts[0]] = 0
    selections = {
        'large_up_all': (delta > 25, np.ones(len(x), dtype=bool)),
        'large_up_normal': ((delta > 25) & (labels == 0), labels == 0),
        'attack_down_gt10pp': ((delta < -10) & (labels == 1), labels == 1),
    }
    tables = {}
    for group, (selected, eligible) in selections.items():
        total = int(selected.sum())
        rows = []
        for i, name in enumerate(names):
            in_bin = bins == i
            mask = selected & in_bin
            count = int(mask.sum())
            base = int((eligible & in_bin).sum())
            rows.append({
                'v2_bin': name, 'count': count,
                'true_attack_prevalence': float(labels[mask].mean()) if count else None,
                'median_v2_margin': float(np.median(x[mask])) if count else None,
                'median_v5f_margin': float(np.median(y[mask])) if count else None,
                'median_rank_shift_pp': float(np.median(delta[mask])) if count else None,
                'fraction_of_selected_group': count / total if total else None,
                'eligible_count_in_bin': base,
                'selected_fraction_of_eligible_in_bin': count / base if base else None,
            })
        assert sum(row['count'] for row in rows) == total
        tables[group] = {'count': total, 'bins': rows}
    return {
        'quantile_population': 'ALL target development rows scored by frozen V2; not adaptation-train quantiles and not subgroup quantiles',
        'v2_score_quantile_thresholds': dict(zip(map(str, levels), cuts.tolist())),
        'bin_policy': 'score <= q02; q02 < score < q25; then [q25,q50), [q50,q75), [q75,q95), [q95,q98), [q98,+inf). Tied q02 scores go to the first bin; ties may cause empty or unequal-sized bins.',
        'denominators': 'fraction_of_selected_group describes origin composition; selected_fraction_of_eligible_in_bin is selection rate among all rows / true Normal / true Attack in that bin, respectively',
        'tables': tables,
    }


def render_origins(result):
    for group, table in result['tables'].items():
        print(f'\n{group}: N={table["count"]}', flush=True)
        print(f'{"V2 bin":<10} {"N":>8} {"Attack prev":>12} {"V2 median":>11} {"V5f median":>11} {"shift pp":>10} {"group share":>12} {"bin rate":>10}')
        for row in table['bins']:
            if not row['count']:
                print(f'{row["v2_bin"]:<10} {0:>8}  --')
                continue
            print(f'{row["v2_bin"]:<10} {row["count"]:>8} '
                  f'{row["true_attack_prevalence"]:>12.2%} '
                  f'{row["median_v2_margin"]:>11.4f} {row["median_v5f_margin"]:>11.4f} '
                  f'{row["median_rank_shift_pp"]:>10.2f} '
                  f'{row["fraction_of_selected_group"]:>12.2%} '
                  f'{row["selected_fraction_of_eligible_in_bin"]:>10.2%}')


def analyze(v2, v5f, labels, parameters, threshold):
    x, y = np.asarray(v2, dtype=np.float64), np.asarray(v5f, dtype=np.float64)
    labels = np.asarray(labels)
    if x.ndim != 1 or x.shape != y.shape or x.shape != labels.shape or len(x) < 2:
        raise ValueError('Need at least two aligned samples')
    if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError('Invalid scores/labels')
    validate_affine(parameters)
    if not math.isfinite(threshold):
        raise ValueError('Nonfinite threshold')
    labels = labels.astype(np.int64)
    rx, ry = rankdata(x, method='average'), rankdata(y, method='average')
    # Subtract ranks before scaling so exact +/-10 and +/-25 boundaries stay exact.
    delta = (ry - rx) * 100 / (len(x) - 1)
    px, py = (rx - 1) * 100 / (len(x) - 1), (ry - 1) * 100 / (len(x) - 1)
    cx, cy = ap_contributions(x, labels), ap_contributions(y, labels)
    predicted = parameters['a'] * y + parameters['b'] >= threshold
    masks = group_masks(delta)
    assert np.all(np.sum(list(masks.values()), axis=0) == 1)
    groups = []
    for name, mask in masks.items():
        n = int(mask.sum())
        row = {'group': name, 'count': n, 'fraction_of_development': n / len(x),
               'true_attack_count': int(labels[mask].sum()),
               'true_attack_prevalence': float(labels[mask].mean()) if n else None,
               'v2_raw_margin': mean_median(x[mask]) if n else None,
               'v5f_raw_margin': mean_median(y[mask]) if n else None,
               'delta_rank_pp': mean_median(delta[mask]) if n else None,
               'v2_global_percentile_rank': mean_median(px[mask]) if n else None,
               'v5f_global_percentile_rank': mean_median(py[mask]) if n else None,
               'v2_descending_rank_position': mean_median(len(x) + 1 - rx[mask]) if n else None,
               'v5f_predicted_attack_rate': float(predicted[mask].mean()) if n else None,
               'v5f_raw_predicted_attack_rate': float((y[mask] >= threshold).mean()) if n else None,
               'v2_ap_contribution': float(cx[mask].sum()) if cx is not None else None,
               'v5f_ap_contribution': float(cy[mask].sum()) if cy is not None else None,
               'ap_contribution_delta': float((cy[mask] - cx[mask]).sum()) if cx is not None else None}
        groups.append(row)
    return {'count': len(x), 'true_attack_prevalence': float(labels.mean()),
            'v2_average_precision': float(cx.sum()) if cx is not None else None,
            'v5f_average_precision': float(cy.sum()) if cy is not None else None,
            'P_attack_given_large_up': groups[-1]['true_attack_prevalence'],
            'P_attack_given_large_down': groups[0]['true_attack_prevalence'],
            'groups': groups,
            'reordering_origins': origin_bins(x, y, labels, delta)}


@torch.no_grad()
def score_development(v2, student, loader, expected):
    old, new, labels = [], [], []
    seen = 0
    for i, (features, batch_labels) in enumerate(loader, 1):
        _, a = v2(features)
        _, b = student(features)
        old.append((a[:, 1] - a[:, 0]).cpu().numpy())
        new.append((b[:, 1] - b[:, 0]).cpu().numpy())
        labels.append(batch_labels.cpu().numpy())
        seen += len(features)
        if i % 200 == 0:
            print(f'Scored {seen:,}/{expected:,} development rows', flush=True)
    if seen != expected or seen < 2:
        raise ValueError(f'Incomplete development scoring: {seen}/{expected}')
    return np.concatenate(old), np.concatenate(new), np.concatenate(labels)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/hda_v5f.json')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', default='results/hda_v5f/diagnostics/development_reordering_seed42.json')
    args = parser.parse_args()
    output = resolve_path(args.output)
    if output.exists():
        parser.error('Output exists; choose a fresh --output')
    config, protocol, provenance, source, v2, teacher = load_context(args.config, args.seed)
    student, checkpoint = load_student(config, provenance, source, teacher)
    verify_training_data(protocol, checkpoint)
    reference = load_v5d_reference(config, protocol, provenance)
    v5b = load_v5b_calibration(config, protocol, provenance)
    artifact = json.loads(calibration_path(config).read_text())
    frozen = artifact['frozen']
    deps = dependencies(config, provenance)
    target = evaluation_target(protocol, 'development')
    snapshot = data_snapshot(target)
    if (payload_hash(frozen) != artifact['sha256'] or frozen['dependencies'] != deps
            or frozen['v5b_calibration_payload_sha256'] != v5b['sha256']
            or frozen['source_validation_files'] != data_snapshot(resolve_path(config['source_validation']))
            or frozen['target_adaptation_files'] != data_snapshot(resolve_path(protocol['target_data']['adaptation_train']))
            or str(target) != frozen['target_development'] or snapshot != reference['development_files']):
        raise ValueError('Frozen calibration, dependencies or development snapshot changed')
    params, threshold = frozen['parameters']['v5f'], frozen['thresholds']['v5f']
    for model in (v2, student):
        model.cpu().eval().requires_grad_(False)
    expected = sum(pq.read_metadata(p).num_rows for p in sorted(target.glob('*.parquet')))
    print(f'Post-hoc DEVELOPMENT only: {expected:,} labeled rows; frozen models and threshold.', flush=True)
    old, new, labels = score_development(v2, student, make_loader(
        target, provenance['target_dim'], protocol['training']['batch_size']), expected)
    if snapshot != data_snapshot(target) or deps != dependencies(config, provenance):
        raise ValueError('Inputs changed during scoring')
    result = analyze(old, new, labels, params, threshold)
    result.update(phase='development', training_seed=config['training_seed'], expected_rows=expected,
                  dependencies=deps, development_path=str(target), development_snapshot=snapshot,
                  calibration_payload_sha256=artifact['sha256'],
                  raw_source_threshold=threshold, calibrated_threshold=threshold,
                  affine={'a': params['a'], 'b': params['b']},
                  rank_definition='Ascending global development average rank for ties; delta_pp=100*(rank_V5f-rank_V2)/(N-1). Positive means higher attack-score rank; descending position 1 means highest score.',
                  prediction_rule='a * V5f_raw_margin + b >= saved source threshold (matches evaluation)',
                  ap_definition='Tie-aware non-interpolated AP. Each positive receives precision at its score threshold / total development positives. Group sums equal global AP; not within-group AP or causal attribution.',
                  note='Post-hoc labeled development diagnostic only. No model training, threshold selection, calibration fitting or final-test access.')
    print(f'Overall attack prevalence: {result["true_attack_prevalence"]:.2%}')
    print(f'V2 AP={result["v2_average_precision"]}; V5f AP={result["v5f_average_precision"]}')
    for row in result['groups']:
        print(json.dumps(row, indent=2), flush=True)
    print('P(Y=1 | delta_rank > 25pp):', result['P_attack_given_large_up'])
    print('P(Y=1 | delta_rank < -25pp):', result['P_attack_given_large_down'])
    render_origins(result['reordering_origins'])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print('Saved:', output)


if __name__ == '__main__':
    main()
