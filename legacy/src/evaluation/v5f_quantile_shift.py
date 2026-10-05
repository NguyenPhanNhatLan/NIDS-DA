"""Score 100% of unlabeled adaptation-train with V2 and V5f, grouped by V2 score."""
import argparse
import json
import math

import numpy as np
import pyarrow.parquet as pq
import torch

from evaluation.hda_v5b_calibration import data_snapshot, payload_hash
from evaluation.hda_v5f import calibration_path, dependencies, load_v5b_calibration, validate_affine, verify_training_data
from evaluation.v5f_pool_classifier import statistics
from training.hda_v5f import load_context, load_student
from training.thesis_protocol import resolve_path
from training.v6_data import make_teacher_loader

LEVELS = [.02, .25, .50, .75, .95, .98]
BINS = [('0-2%', 'pseudo-Normal anchor'), ('2-25%', 'low-score'),
        ('25-50%', 'mid-low'), ('50-75%', 'mid-high'), ('75-95%', 'high'),
        ('95-98%', 'pseudo-Attack anchor'), ('98-100%', 'extreme high')]


def summarize_bins(v2, v5f, parameters, threshold):
    if v2.ndim != 1 or v2.shape != v5f.shape or not len(v2):
        raise ValueError('Expected nonempty paired margin vectors')
    if not torch.isfinite(v2).all() or not torch.isfinite(v5f).all():
        raise ValueError('Nonfinite margins')
    # Match the original NumPy linear quantiles used to create pseudo pools.
    cuts = np.quantile(v2.numpy(), LEVELS)
    if cuts[0] >= cuts[4]:
        raise ValueError('q02 >= q95: tied scores prevent distinct Normal/Attack anchor bins')
    assignments = np.searchsorted(cuts, v2.numpy(), side='right')
    assignments[v2.numpy() <= cuts[0]] = 0
    rows = []
    for index, (label, meaning) in enumerate(BINS):
        mask = torch.from_numpy(assignments == index)
        n = int(mask.sum())
        row = {'bin': label, 'meaning': meaning, 'count': n, 'fraction_of_total': n / len(v2)}
        if n:
            old, new = v2[mask].double(), v5f[mask].double()
            delta = new - old
            row.update(v2_margin=statistics(old), v5f_margin=statistics(new),
                       delta_margin=statistics(delta),
                       fraction_v5f_margin_gt_zero=(new > 0).double().mean().item(),
                       fraction_v5f_margin_gt_raw_source_threshold=(new > threshold).double().mean().item(),
                       fraction_v5f_calibrated_margin_gt_threshold=(
                           parameters['a'] * new + parameters['b'] > threshold).double().mean().item(),
                       fraction_delta_positive=(delta > 0).double().mean().item())
        else:
            row.update(v2_margin=None, v5f_margin=None, delta_margin=None,
                       fraction_v5f_margin_gt_zero=None,
                       fraction_v5f_margin_gt_raw_source_threshold=None,
                       fraction_v5f_calibrated_margin_gt_threshold=None,
                       fraction_delta_positive=None)
        rows.append(row)
    assert sum(row['count'] for row in rows) == len(v2)
    return dict(zip([str(q) for q in LEVELS], cuts.tolist())), rows


@torch.no_grad()
def score_all(v2, student, loader, expected_rows):
    old, new = [], []
    seen = 0
    for i, x in enumerate(loader, 1):
        _, v2_logits = v2(x)
        _, v5f_logits = student(x)
        old.append((v2_logits[:, 1] - v2_logits[:, 0]).cpu())
        new.append((v5f_logits[:, 1] - v5f_logits[:, 0]).cpu())
        seen += len(x)
        if i % 200 == 0:
            print(f'Scored {seen:,}/{expected_rows:,} rows ({seen / expected_rows:.1%})', flush=True)
    if seen != expected_rows or seen == 0:
        raise ValueError(f'Incomplete scoring: {seen} / {expected_rows}')
    return torch.cat(old), torch.cat(new)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/hda_v5f.json')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', default='results/hda_v5f/diagnostics/quantile_shift_seed42.json')
    args = parser.parse_args()
    output = resolve_path(args.output)
    if output.exists():
        parser.error('Output exists; choose a fresh --output')
    config, protocol, provenance, source, v2, teacher = load_context(args.config, args.seed)
    student, checkpoint = load_student(config, provenance, source, teacher)
    verify_training_data(protocol, checkpoint)
    v5b = load_v5b_calibration(config, protocol, provenance)
    path = calibration_path(config)
    artifact = json.loads(path.read_text())
    frozen = artifact['frozen']
    deps = dependencies(config, provenance)
    target_path = resolve_path(protocol['target_data']['adaptation_train'])
    snapshot = data_snapshot(target_path)
    if (payload_hash(frozen) != artifact['sha256'] or frozen['dependencies'] != deps
            or frozen['v5b_calibration_payload_sha256'] != v5b['sha256']
            or frozen['source_validation_files'] != data_snapshot(resolve_path(config['source_validation']))
            or frozen['target_adaptation_files'] != snapshot):
        raise ValueError('Frozen calibration, dependencies or fitting data changed')
    params, threshold = frozen['parameters']['v5f'], frozen['thresholds']['v5f']
    validate_affine(params)
    if not math.isfinite(threshold):
        raise ValueError('Nonfinite threshold')
    for model in (v2, student):
        model.cpu().eval().requires_grad_(False)
    expected = sum(pq.read_metadata(p).num_rows for p in sorted(target_path.glob('*.parquet')))
    print(f'Scoring all {expected:,} adaptation-train rows on CPU; target labels are not loaded.', flush=True)
    old, new = score_all(v2, student, make_teacher_loader(
        target_path, provenance['target_dim'], protocol['training']['batch_size']), expected)
    if data_snapshot(target_path) != snapshot or dependencies(config, provenance) != deps:
        raise ValueError('Inputs changed during scoring')
    cuts, rows = summarize_bins(old, new, params, threshold)
    result = {'training_seed': config['training_seed'], 'device': 'cpu',
              'dependencies': deps, 'calibration_payload_sha256': artifact['sha256'],
              'target_path': str(target_path), 'target_snapshot': snapshot,
              'target_labels_used': False, 'expected_rows': expected, 'scored_rows': len(old),
              'quantile_thresholds': cuts, 'raw_source_threshold': threshold,
              'calibrated_threshold': threshold, 'affine': {'a': params['a'], 'b': params['b']},
              'raw_equivalent_calibrated_threshold': (threshold - params['b']) / params['a'],
              'bin_policy': 'score <= q02; q02 < score < q25; then [q25,q50), [q50,q75), [q75,q95), [q95,q98), [q98,+inf). q02 ties belong to Normal. Ties can change bin sizes; empty bins have null statistics.',
              'delta_definition': 'paired raw V5f margin minus raw V2 margin, before affine calibration',
              'note': 'Descriptive adaptation-train diagnostic; different classifiers can have different score scales. Not target-label accuracy.',
              'bins': rows}
    print(f'Raw source threshold t={threshold:.8f}; calibrated margin={params["a"]:.8f}*m+({params["b"]:.8f})')
    print(f'{"V2 bin":<10} {"N":>9} {"V2 med":>10} {"V5f med":>10} {"P(m>0)":>10} {"P(m>t)":>10} {"P(cal>t)":>10} {"delta med":>11} {"delta mean":>11}')
    for row in rows:
        if not row['count']:
            print(row['bin'], 'EMPTY')
            continue
        print(f'{row["bin"]:<10} {row["count"]:>9} {row["v2_margin"]["median"]:>10.4f} '
              f'{row["v5f_margin"]["median"]:>10.4f} {row["fraction_v5f_margin_gt_zero"]:>10.2%} '
              f'{row["fraction_v5f_margin_gt_raw_source_threshold"]:>10.2%} '
              f'{row["fraction_v5f_calibrated_margin_gt_threshold"]:>10.2%} '
              f'{row["delta_margin"]["median"]:>11.4f} {row["delta_margin"]["mean"]:>11.4f}')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print('Saved:', output)


if __name__ == '__main__':
    main()
