"""Full adaptation-train V2 -> V5f score correlation, affine fit and rank movement."""
import argparse
import json

import numpy as np
import pyarrow.parquet as pq
from scipy.stats import rankdata

from evaluation.hda_v5b_calibration import data_snapshot
from evaluation.hda_v5f import dependencies, verify_training_data
from evaluation.v5f_quantile_shift import BINS, LEVELS, score_all
from training.hda_v5f import load_context, load_student
from training.thesis_protocol import resolve_path
from training.v6_data import make_teacher_loader


def correlation(x, y):
    dx, dy = x - x.mean(), y - y.mean()
    nx, ny = np.linalg.norm(dx), np.linalg.norm(dy)
    if nx == 0 or ny == 0:
        return None
    return float(np.clip(np.dot(dx / nx, dy / ny), -1., 1.))


def distribution(x):
    names = ('min', 'q05', 'q25', 'median', 'q75', 'q95', 'max')
    return {**dict(zip(names, np.quantile(x, [0, .05, .25, .5, .75, .95, 1]).tolist())),
            'mean': float(x.mean()), 'std_population': float(x.std())}


def movements(delta_pp):
    return {'signed_percentile_points': distribution(delta_pp),
            'absolute_percentile_points': distribution(np.abs(delta_pp)),
            'fraction_up': float(np.mean(delta_pp > 0)),
            'fraction_down': float(np.mean(delta_pp < 0)),
            'fraction_unchanged': float(np.mean(delta_pp == 0)),
            'fraction_absolute_movement_gt_pp': {
                str(t): float(np.mean(np.abs(delta_pp) > t)) for t in (1, 5, 10, 25)}}


def analyze(v2, v5f):
    x, y = np.asarray(v2, dtype=np.float64), np.asarray(v5f, dtype=np.float64)
    if x.ndim != 1 or x.shape != y.shape or len(x) < 2:
        raise ValueError('Need at least two paired one-dimensional margin values')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Nonfinite margins')
    rx, ry = rankdata(x, method='average'), rankdata(y, method='average')
    px, py = (rx - 1) / (len(x) - 1), (ry - 1) / (len(x) - 1)
    delta = 100 * (py - px)
    xc, yc = x - x.mean(), y - y.mean()
    ssx, ssy = float(np.dot(xc, xc)), float(np.dot(yc, yc))
    slope = float(np.dot(xc, yc) / ssx) if ssx > 0 else None
    intercept = float(y.mean() - slope * x.mean()) if slope is not None else None
    residual = yc - slope * xc if slope is not None else yc
    sse = float(np.dot(residual, residual))
    fit = {'definition': 'unconstrained OLS: raw V5f margin = a * raw V2 margin + b; fitted and scored on all rows',
           'a': slope, 'b': intercept, 'r_squared': 1 - sse / ssy if ssy > 0 and ssx > 0 else None,
           'rmse': float(np.sqrt(np.mean(residual ** 2))),
           'residual': distribution(residual),
           'undefined_reason': 'constant V2 or V5f margins' if ssx == 0 or ssy == 0 else None}
    # Percentile ranks, not raw-score quantile thresholds: ties get their average rank.
    cuts = np.asarray(LEVELS)
    bx, by = np.searchsorted(cuts, px, side='right'), np.searchsorted(cuts, py, side='right')
    bx[px <= .02] = 0
    by[py <= .02] = 0
    matrix = np.zeros((7, 7), dtype=np.int64)
    np.add.at(matrix, (bx, by), 1)
    rows = []
    for i, (label, meaning) in enumerate(BINS):
        mask = bx == i
        n = int(mask.sum())
        rows.append({'v2_rank_bin': label, 'meaning': meaning, 'count': n,
                     'rank_movement': movements(delta[mask]) if n else None,
                     'v2_margin_median': float(np.median(x[mask])) if n else None,
                     'v5f_margin_median': float(np.median(y[mask])) if n else None})
    return {'count': len(x), 'pearson': correlation(x, y), 'spearman': correlation(rx, ry),
            'correlation_undefined_policy': 'null when either vector is constant',
            'affine_fit': fit,
            'rank_definition': 'ascending average ranks for ties; percentile=(rank-1)/(N-1); movement=100*(V5f percentile - V2 percentile). Positive means higher attack-score rank.',
            'rank_movement': movements(delta),
            'rank_transition': {'labels': [b[0] for b in BINS],
                                'definition': 'rows=V2 global percentile rank; columns=V5f global percentile rank. <=2%, (2%,25%), then left-closed intervals; ties use average rank, so these bins can differ from raw-score anchor pools.',
                                'counts': matrix.tolist(),
                                'row_fractions': [(row / row.sum()).tolist() if row.sum() else [None] * 7 for row in matrix]},
            'by_v2_rank_bin': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/hda_v5f.json')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', default='results/hda_v5f/diagnostics/rank_movement_seed42.json')
    args = parser.parse_args()
    output = resolve_path(args.output)
    if output.exists():
        parser.error('Output exists; choose a fresh --output')
    config, protocol, provenance, source, v2, teacher = load_context(args.config, args.seed)
    student, checkpoint = load_student(config, provenance, source, teacher)
    verify_training_data(protocol, checkpoint)
    deps = dependencies(config, provenance)
    target = resolve_path(protocol['target_data']['adaptation_train'])
    snapshot = data_snapshot(target)
    expected = sum(pq.read_metadata(p).num_rows for p in sorted(target.glob('*.parquet')))
    for model in (v2, student):
        model.cpu().eval().requires_grad_(False)
    print(f'Scoring 100% adaptation-train: {expected:,} rows on CPU; no target labels.', flush=True)
    old, new = score_all(v2, student, make_teacher_loader(
        target, provenance['target_dim'], protocol['training']['batch_size']), expected)
    if data_snapshot(target) != snapshot or dependencies(config, provenance) != deps:
        raise ValueError('Inputs changed during scoring')
    result = analyze(old.numpy(), new.numpy())
    result.update(training_seed=config['training_seed'], dependencies=deps,
                  expected_rows=expected, scored_rows=len(old), target_path=str(target),
                  target_snapshot=snapshot, target_labels_used=False, device='cpu',
                  note='Descriptive paired raw-margin diagnostic. OLS affine fit is not the evaluation calibrator; no fitting artifacts or model weights are changed.')
    print(f'Pearson: {result["pearson"]}\nSpearman: {result["spearman"]}')
    print('Affine fit:', json.dumps(result['affine_fit'], indent=2))
    print('Global rank movement:', json.dumps(result['rank_movement'], indent=2))
    print(f'{"V2 rank bin":<12} {"N":>9} {"median delta pp":>17} {"median abs pp":>15} {"P(|delta|>10pp)":>18}')
    for row in result['by_v2_rank_bin']:
        m = row['rank_movement']
        if m is None:
            print(row['v2_rank_bin'], 'EMPTY')
        else:
            print(f'{row["v2_rank_bin"]:<12} {row["count"]:>9} '
                  f'{m["signed_percentile_points"]["median"]:>17.4f} '
                  f'{m["absolute_percentile_points"]["median"]:>15.4f} '
                  f'{m["fraction_absolute_movement_gt_pp"]["10"]:>18.2%}')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print('Saved:', output)


if __name__ == '__main__':
    main()
