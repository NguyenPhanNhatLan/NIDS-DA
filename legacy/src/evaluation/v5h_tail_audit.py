"""Post-hoc V5f/V5h development ranking audit; no model or operating-threshold fitting."""
import argparse
import json

import numpy as np
import pyarrow.parquet as pq
import torch
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from evaluation.hda_v5b_calibration import data_snapshot
from evaluation.hda_v5f import verify_training_data
from evaluation.v5f_development_reordering import ap_contributions
from training import hda_v5f, hda_v5h
from training.hda_v5b import file_hash
from training.thesis_protocol import evaluation_target, resolve_path
from training.v6_data import make_loader

TOP = (.01, .02, .05, .10, .20)
FPRS = (.01, .02, .03, .05)


def class_counts(mask, labels):
    return {'count': int(mask.sum()), 'attack': int(labels[mask].sum()),
            'normal': int(mask.sum() - labels[mask].sum())}


def top_statistics(scores, labels):
    order = np.argsort(-scores, kind='stable')
    contributions = ap_contributions(scores, labels)
    rows, masks = {}, {}
    for fraction in TOP:
        k = int(np.ceil(len(scores) * fraction))
        mask = np.zeros(len(scores), dtype=bool)
        mask[order[:k]] = True
        cutoff = scores[order[k-1]]
        tied = scores == cutoff
        inclusive = scores >= cutoff
        key = f'{fraction:.0%}'
        masks[key] = mask
        rows[key] = {'k': k, **class_counts(mask, labels),
                     'attack_prevalence': float(labels[mask].mean()),
                     'precision_at_k': float(labels[mask].mean()),
                     'recall_at_k': float(labels[mask].sum()/labels.sum()),
                     'score_cutoff': float(cutoff),
                     'cutoff_tied_rows': int(tied.sum()), 'cutoff_tied_rows_selected': int((tied & mask).sum()),
                     'inclusive_ties_count': int(inclusive.sum()),
                     'inclusive_ties_precision': float(labels[inclusive].mean()),
                     'global_ap_contribution': float(contributions[mask].sum())}
    return rows, masks


def recall_at_fpr(scores, labels):
    fpr, tpr, thresholds = roc_curve(labels, scores, drop_intermediate=False)
    rows = {}
    for budget in FPRS:
        candidates = np.flatnonzero(fpr <= budget)
        # Max empirical recall without exceeding budget; lowest FPR breaks ties.
        index = candidates[np.argmax(tpr[candidates])]
        threshold = thresholds[index]
        predictions = scores >= threshold
        rows[f'{budget:.0%}'] = {'target_fpr': budget, 'achieved_fpr': float(fpr[index]),
                               'recall': float(tpr[index]),
                               'development_diagnostic_threshold': float(threshold) if np.isfinite(threshold) else None,
                               'predict_none': bool(not predictions.any()),
                               **class_counts(predictions, labels)}
    return rows


def analyze(old, new, labels):
    old, new = np.asarray(old, dtype=float), np.asarray(new, dtype=float)
    labels = np.asarray(labels)
    if old.ndim != 1 or old.shape != new.shape or old.shape != labels.shape or len(old) < 2:
        raise ValueError('Need paired one-dimensional margins and labels')
    if not np.isfinite(old).all() or not np.isfinite(new).all() or set(np.unique(labels)) != {0, 1}:
        raise ValueError('Need finite margins and both binary development classes')
    labels = labels.astype(int)
    delta = 100 * (rankdata(new, method='average') - rankdata(old, method='average')) / (len(old)-1)
    models, masks = {}, {}
    for name, scores in (('v5f', old), ('v5h', new)):
        top, masks[name] = top_statistics(scores, labels)
        models[name] = {'roc_auc': float(roc_auc_score(labels, scores)),
                        'average_precision': float(average_precision_score(labels, scores)),
                        'top_score': top, 'recall_at_target_fpr': recall_at_fpr(scores, labels)}
    movement = {name: class_counts(mask, labels) for name, mask in {
        'up': delta > 0, 'down': delta < 0, 'unchanged': delta == 0,
        'up_gt10pp': delta > 10, 'down_gt10pp': delta < -10,
        'up_gt25pp': delta > 25, 'down_gt25pp': delta < -25}.items()}
    changes = {}
    for key in masks['v5f']:
        f, h = masks['v5f'][key], masks['v5h'][key]
        changes[key] = {'entered_v5h_top': class_counts(h & ~f, labels),
                        'left_v5f_top': class_counts(f & ~h, labels),
                        'precision_delta': models['v5h']['top_score'][key]['precision_at_k'] - models['v5f']['top_score'][key]['precision_at_k']}
    return {'count': len(labels), 'attack_prevalence': float(labels.mean()), 'models': models,
            'delta_v5h_minus_v5f': {key: models['v5h'][key] - models['v5f'][key] for key in ('roc_auc', 'average_precision')},
            'rank_movement': movement, 'top_membership_changes': changes,
            'definitions': {
                'scores': 'raw classifier margin logit_attack - logit_normal; positive affine does not change ranks',
                'top_k': 'k=ceil(N*fraction); descending scores, stable original row order breaks cutoff ties; inclusive-ties precision also reported',
                'precision': 'Attack prevalence in top-k equals precision@k',
                'rank_movement': 'ascending average rank for ties; delta_pp=100*(rank_V5h-rank_V5f)/(N-1); positive=up',
                'target_fpr': 'best empirical recall among whole-score thresholds with achieved FPR <= budget; no interpolation; descriptive development ROC operating points only',
                'ap_contribution': 'sum of per-positive precision-at-score-threshold / total positives within top-k; nested top-k values overlap',
            }}


@torch.no_grad()
def score_pair(f, h, loader, expected):
    old, new, labels = [], [], []
    seen = 0
    for step, (x, y) in enumerate(loader, 1):
        _, a = f(x)
        _, b = h(x)
        old.append((a[:,1]-a[:,0]).numpy())
        new.append((b[:,1]-b[:,0]).numpy())
        labels.append(y.numpy())
        seen += len(x)
        if step % 200 == 0:
            print(f'Development scored: {seen:,}/{expected:,}', flush=True)
    if seen != expected or seen == 0:
        raise ValueError(f'Incomplete scoring: {seen}/{expected}')
    return np.concatenate(old), np.concatenate(new), np.concatenate(labels)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--v5f-config', default='configs/hda_v5f.json')
    parser.add_argument('--v5h-config', default='configs/hda_v5h.json')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', default='results/hda_v5h/diagnostics/tail_vs_v5f_seed42.json')
    args = parser.parse_args()
    output = resolve_path(args.output)
    if output.exists():
        parser.error('Output exists; choose a fresh --output')
    models, dependencies, contexts = {}, {}, {}
    for name, module, config_path in (('v5f', hda_v5f, args.v5f_config), ('v5h', hda_v5h, args.v5h_config)):
        config, protocol, provenance, source, _, teacher = module.load_context(config_path, args.seed)
        model, checkpoint = module.load_student(config, provenance, source, teacher)
        verify_training_data(protocol, checkpoint)
        reference = hda_v5f.load_v5d_reference(config, protocol, provenance)
        models[name] = model.cpu().eval().requires_grad_(False)
        dependencies[name] = {'checkpoint': str(module.checkpoint_path(config)),
                              'checkpoint_sha256': file_hash(module.checkpoint_path(config)),
                              'provenance': provenance, 'code_sha256': module.code_hashes()}
        contexts[name] = (protocol, provenance, reference)
    protocol, provenance, reference = contexts['v5f']
    target = evaluation_target(protocol, 'development')
    other_protocol, other_provenance, other_reference = contexts['v5h']
    if target.resolve() != evaluation_target(other_protocol, 'development').resolve() or provenance['target_dim'] != other_provenance['target_dim']:
        raise ValueError('Models do not use the same development data')
    snapshot = data_snapshot(target)
    if snapshot != reference['development_files'] or snapshot != other_reference['development_files']:
        raise ValueError('Development snapshot differs from pinned references')
    expected = sum(pq.read_metadata(p).num_rows for p in sorted(target.glob('*.parquet')))
    print(f'Frozen V5f/V5h: scoring all {expected:,} DEVELOPMENT rows on CPU.', flush=True)
    old, new, labels = score_pair(models['v5f'], models['v5h'], make_loader(
        target, provenance['target_dim'], protocol['training']['batch_size']), expected)
    if snapshot != data_snapshot(target):
        raise ValueError('Development data changed during scoring')
    for name, module in (('v5f', hda_v5f), ('v5h', hda_v5h)):
        dep = dependencies[name]
        if file_hash(resolve_path(dep['checkpoint'])) != dep['checkpoint_sha256'] or module.code_hashes() != dep['code_sha256']:
            raise ValueError('Checkpoint or code changed during diagnostic')
    result = analyze(old, new, labels)
    result.update(phase='post-hoc development', training_seed=args.seed, dependencies=dependencies,
                  development_path=str(target), development_snapshot=snapshot, expected_rows=expected,
                  note='No model/calibration fitting, saved operating-threshold changes or final-test access. Tail deterioration is descriptive evidence, not proof that a particular loss term caused it.')
    print('V5h - V5f:', result['delta_v5h_minus_v5f'])
    print('Top score | V5f precision | V5h precision | delta')
    for key, change in result['top_membership_changes'].items():
        print(key, *(f'{result["models"][name]["top_score"][key]["precision_at_k"]:.2%}' for name in ('v5f','v5h')), f'{change["precision_delta"]:+.2%}')
    for name in ('v5f','v5h'):
        print(name, 'recall@target-FPR:', json.dumps(result['models'][name]['recall_at_target_fpr'], indent=2))
    print('Ranking changes (Attack/Normal):', json.dumps(result['rank_movement'], indent=2))
    print('Top membership changes:', json.dumps(result['top_membership_changes'], indent=2))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print('Saved:', output)


if __name__ == '__main__':
    main()
