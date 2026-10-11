"""Versioned E1/E2 extension; frozen proposal_v2 artifacts are read-only inputs."""
import argparse
from copy import deepcopy
import csv
import json
import math
from pathlib import Path

import torch
from torch import nn

from evaluation.baseline import collect_scores, compute_metrics, evaluate_ap, select_f1_threshold
from features.common_features import COMMON_FEATURES
from models.baseline import BaselineMLP
from training.baseline import set_seed
from training.data_revision import file_sha256, require_development_open, revision_path, verify_revision
from training.proposal_data import ParquetBatchStream, split_sha256
from training.proposal_mkmmd import multi_kernel_mmd_loss
from training.supcon_extension import supervised_contrastive_loss

ROOT = Path(__file__).resolve().parents[2]
VERSION = 'supcon_mkmmd_ext_v1'
DEFAULT_CONFIG = ROOT / 'configs/proposal_supcon_mkmmd_ext_v1.json'
EXPERIMENTS = ('E1', 'E2')
DIRECTIONS = ('unsw_to_cicids', 'cicids_to_unsw')
METRICS = ('pr_auc', 'roc_auc', 'macro_f1', 'recall', 'fpr')


def load_config(path):
    config = json.loads(path.read_text(encoding='utf-8'))
    fixed = {
        'extension_id': VERSION, 'architecture': 'BaselineMLP',
        'feature_count': len(COMMON_FEATURES), 'latent_dim': 168,
        'experiments': list(EXPERIMENTS),
        'initialization': 'corresponding_source_only_checkpoint_independent',
        'supcon_representation': 'l2_normalized_source_latent',
        'alignment_representation': 'raw_168d_latent',
        'checkpoint_selection': 'source_val_ap_including_epoch0',
        'threshold_selection': 'source_val_f1', 'bn_running_stats_frozen': True,
        'target_labels_used_training': False,
    }
    for key, value in fixed.items():
        if config.get(key) != value:
            raise ValueError(f'Unsupported extension setting: {key}')
    settings = config['training']
    for key, value in ((key, config[key]) for key in ('alpha_ce', 'alpha_supcon', 'temperature', 'lambda_mkmmd_e2')):
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'{key} must be finite and positive')
    scales = config['kernel_scales']
    if not scales or any(not math.isfinite(s) or s <= 0 for s in scales):
        raise ValueError('Kernel scales must be finite and positive')
    for key, minimum in (('batch_size', 2), ('epochs', 1), ('patience', 1)):
        if type(settings[key]) is not int or settings[key] < minimum:
            raise ValueError(f'Invalid training {key}')
    for key in ('learning_rate', 'weight_decay', 'min_delta'):
        if not math.isfinite(settings[key]) or settings[key] < 0:
            raise ValueError(f'Invalid training {key}')
    if settings['learning_rate'] == 0:
        raise ValueError('Learning rate must be positive')
    return config


def artifact_paths(direction, seed, experiment):
    model = revision_path('model_root', None) / 'extensions' / VERSION / experiment / direction / f'seed{seed}.pt'
    result = revision_path('result_root', None) / 'extensions' / VERSION / experiment / direction / f'seed{seed}.json'
    return model, result


def reference(direction, seed):
    # Existing validator replays source AP/threshold and checks all frozen provenance.
    # It never reads target-test labels or selects on target validation.
    from evaluation.proposal_final_test import validate_development
    validated = validate_development(direction, 'source_only', seed)
    if not validated['revision']:
        raise ValueError('A frozen data revision is required')
    checkpoint = torch.load(validated['checkpoint_path'], map_location='cpu', weights_only=True)
    result = json.loads(validated['result_path'].read_text(encoding='utf-8'))
    return validated, checkpoint, result


def provenance(validated, checkpoint, direction, seed, config_path):
    return {
        **validated['revision'], 'extension_id': VERSION, 'architecture': 'BaselineMLP',
        'direction': direction, 'seed': seed, 'features': list(COMMON_FEATURES),
        'input_dim': len(COMMON_FEATURES), 'latent_dim': 168,
        'config_sha256': file_sha256(config_path),
        'source_checkpoint': str(validated['checkpoint_path']),
        'source_checkpoint_sha256': file_sha256(validated['checkpoint_path']),
        'source_result_sha256': file_sha256(validated['result_path']),
        'prepared_split_sha256': checkpoint['prepared_split_sha256'],
        'preprocessor_sha256': checkpoint['preprocessor_sha256'],
        'common_feature_config_sha256': checkpoint['common_feature_config_sha256'],
    }


def train_extension(model, source, target, validation, counts, config, experiment, initial_ap):
    from training.proposal_mmd import freeze_bn_stats
    device = next(model.parameters()).device
    settings = config['training']
    counts = torch.as_tensor(counts, dtype=torch.float32, device=device)
    if counts.shape != (2,) or (counts <= 0).any():
        raise ValueError('Source must contain both classes')
    criterion = nn.CrossEntropyLoss(weight=counts.sum() / (2 * counts))
    optimizer = torch.optim.Adam(model.parameters(), lr=settings['learning_rate'], weight_decay=settings['weight_decay'])
    initial = evaluate_ap(model, validation, device)
    if not math.isclose(initial, initial_ap, rel_tol=0, abs_tol=1e-3):
        raise ValueError('Source checkpoint AP does not replay')
    best_ap, best_epoch = float(initial), 0
    best_state = deepcopy(model.state_dict())
    history = [{'epoch': 0, 'stage': 'source_pretrained', 'source_val_ap': best_ap}]
    stale = 0
    weight = config['lambda_mkmmd_e2'] if experiment == 'E2' else 0.0
    for epoch in range(1, settings['epochs'] + 1):
        model.train()
        freeze_bn_stats(model)
        iterator = iter(target) if target is not None else None
        totals = {key: 0.0 for key in ('loss', 'source_ce', 'source_supcon', 'mkmmd', 'bandwidth')}
        steps = valid_anchors = source_rows = zero_positive_batches = aligned = 0
        for x, labels in source:
            x, labels = x.to(device), labels.to(device)
            # Identical source forward order in E1/E2. Target dropout RNG is isolated.
            z, logits = model(x)
            ce = criterion(logits, labels)
            contrastive, anchors = supervised_contrastive_loss(z, labels, config['temperature'])
            mmd, bandwidth = z.new_zeros(()), z.new_zeros(())
            if weight:
                try:
                    tx = next(iterator)
                except StopIteration:
                    iterator = iter(target)
                    try:
                        tx = next(iterator)
                    except StopIteration:
                        raise ValueError('Target has no full training batch') from None
                if len(tx) != len(x):
                    raise ValueError('Source and target batch sizes must match')
                cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
                with torch.random.fork_rng(devices=cuda_devices):
                    target_z, _ = model(tx.to(device))
                mmd, bandwidth = multi_kernel_mmd_loss(z, target_z, config['kernel_scales'])
                aligned += 1
            loss = config['alpha_ce'] * ce + config['alpha_supcon'] * contrastive + weight * mmd
            if not torch.isfinite(loss):
                raise ValueError('Non-finite extension training loss')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            for key, value in (('loss', loss), ('source_ce', ce), ('source_supcon', contrastive), ('mkmmd', mmd), ('bandwidth', bandwidth)):
                totals[key] += float(value.detach())
            steps += 1
            source_rows += len(x)
            valid_anchors += anchors
            zero_positive_batches += int(anchors == 0)
        if not steps:
            raise ValueError('Source has no full training batch')
        ap = float(evaluate_ap(model, validation, device))
        if not math.isfinite(ap):
            raise ValueError('Non-finite source validation AP')
        row = {'epoch': epoch, 'source_val_ap': ap, 'lambda_mkmmd': weight,
               **{key: value / steps for key, value in totals.items()},
               'batches': steps, 'aligned_batches': aligned,
               'source_rows': source_rows, 'supcon_valid_anchors': valid_anchors,
               'supcon_valid_anchor_fraction': valid_anchors / source_rows,
               'supcon_no_positive_batches': zero_positive_batches}
        history.append(row)
        print(f"{experiment} epoch {epoch:03d}: CE={row['source_ce']:.5f}, SupCon={row['source_supcon']:.5f}, MK-MMD={row['mkmmd']:.5f}, source AP={ap:.6f}")
        improvement = ap - best_ap
        if ap > best_ap:
            best_ap, best_epoch, best_state = ap, epoch, deepcopy(model.state_dict())
        stale = 0 if improvement > settings['min_delta'] else stale + 1
        if stale >= settings['patience']:
            break
    model.load_state_dict(best_state)
    model.eval()
    return model, best_epoch, best_ap, history


def train(direction, seed, experiment, config_path):
    require_development_open()
    config = load_config(config_path)
    if not verify_revision():
        raise ValueError('A frozen data revision is required')
    model_path, result_path = artifact_paths(direction, seed, experiment)
    if model_path.exists() or result_path.exists():
        raise FileExistsError(f'Existing or incomplete run: {model_path}, {result_path}')
    validated, checkpoint, baseline = reference(direction, seed)
    meta = provenance(validated, checkpoint, direction, seed, config_path)
    source_domain = 'unsw' if direction == 'unsw_to_cicids' else 'cicids'
    base = validated['base']
    counts = torch.zeros(2, dtype=torch.long)
    for _, y in ParquetBatchStream(base / f'{source_domain}_train', 4096, False, seed, True):
        counts += torch.bincount(y, minlength=2)
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # New instance/optimizer for each experiment; E2 never loads E1.
    model = BaselineMLP(len(COMMON_FEATURES)).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    batch = config['training']['batch_size']
    source = ParquetBatchStream(base / f'{source_domain}_train', batch, True, seed, True, drop_last=True)
    target = (ParquetBatchStream(base / f"{validated['target']}_train", batch, True, seed + 1000, False, drop_last=True)
              if experiment == 'E2' else None)
    validation = ParquetBatchStream(base / f'{source_domain}_val', 1024, False, seed, True)
    model, epoch, ap, history = train_extension(model, source, target, validation, counts.tolist(), config,
                                               experiment, validated['source_val_ap'])
    y, s = collect_scores(model, validation)
    threshold = select_f1_threshold(y, s)
    meta.update(experiment=experiment, method='ce_supcon' if experiment == 'E1' else 'ce_supcon_mkmmd',
                best_epoch=epoch, best_source_val_ap=ap,
                selected_stage='source_pretrained' if epoch == 0 else 'adaptation',
                threshold=threshold, checkpoint_selection=config['checkpoint_selection'],
                threshold_selection=config['threshold_selection'], bn_running_stats_frozen=True,
                target_labels_used_training=False, target_labels_used_checkpoint_selection=False,
                target_labels_used_threshold_selection=False, config=config, history=history,
                source_train_counts=counts.tolist())
    # Target labels are used only after source-only checkpoint and threshold selection.
    ty, ts = collect_scores(model, ParquetBatchStream(base / f"{validated['target']}_val", 1024, False, seed, True))
    target_metrics = compute_metrics(ty, ts, threshold)
    result = {**meta, 'checkpoint': str(model_path), 'data_role': 'target_validation_development_only',
              'source_val': compute_metrics(y, s, threshold), 'target_development': target_metrics,
              'target_development_split': f"{validated['target']}_val",
              'source_only_reference': baseline['target_development'],
              'delta_vs_source_only': {key: target_metrics[key] - baseline['target_development'][key] for key in METRICS}}
    model_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open('xb') as stream:
        torch.save({**meta, 'model_state_dict': {key: value.detach().cpu() for key, value in model.state_dict().items()}}, stream)
    result['checkpoint_sha256'] = file_sha256(model_path)
    with result_path.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(f'Saved: {result_path}')


def validate_run(direction, seed, experiment, config_path, validated, source_checkpoint):
    model_path, result_path = artifact_paths(direction, seed, experiment)
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
    result = json.loads(result_path.read_text(encoding='utf-8'))
    expected = {**provenance(validated, source_checkpoint, direction, seed, config_path),
                'experiment': experiment,
                'method': 'ce_supcon' if experiment == 'E1' else 'ce_supcon_mkmmd',
                'target_labels_used_training': False,
                'target_labels_used_checkpoint_selection': False,
                'target_labels_used_threshold_selection': False}
    for artifact in (checkpoint, result):
        if any(artifact.get(key) != value for key, value in expected.items()):
            raise ValueError(f'Extension provenance mismatch: {experiment}, seed{seed}')
    if result['checkpoint'] != str(model_path) or result['checkpoint_sha256'] != file_sha256(model_path):
        raise ValueError('Extension checkpoint/result identity mismatch')
    for key in ('best_epoch', 'best_source_val_ap', 'threshold', 'selected_stage'):
        if checkpoint[key] != result[key]:
            raise ValueError(f'Extension selection mismatch: {key}')
    model = BaselineMLP(len(COMMON_FEATURES))
    model.load_state_dict(checkpoint['model_state_dict'])
    source = 'unsw' if direction == 'unsw_to_cicids' else 'cicids'
    y, s = collect_scores(model, ParquetBatchStream(validated['base'] / f'{source}_val', 1024, False, seed, True))
    metrics = compute_metrics(y, s, checkpoint['threshold'])
    if (not math.isclose(metrics['pr_auc'], checkpoint['best_source_val_ap'], abs_tol=1e-3, rel_tol=0)
            or not math.isclose(select_f1_threshold(y, s), checkpoint['threshold'], abs_tol=1e-6, rel_tol=0)):
        raise ValueError('Extension source AP/threshold replay differs')
    return model, result, metrics


def evaluate(direction, seed, config_path, output_name):
    from evaluation.proposal_score_diagnostic import score_diagnostics
    load_config(config_path)
    validated, checkpoint, baseline = reference(direction, seed)
    output = revision_path('result_root', None) / 'extensions' / VERSION / output_name / direction / f'seed{seed}.json'
    if output.exists():
        raise FileExistsError(f'Evaluation already exists: {output}')
    checked = {exp: validate_run(direction, seed, exp, config_path, validated, checkpoint) for exp in EXPERIMENTS}
    entries = {'source_only': (validated['model'], validated['threshold'], None)}
    entries.update({exp: (model, result['threshold'], result) for exp, (model, result, _) in checked.items()})
    runs = {}
    for name, (model, threshold, result) in entries.items():
        y, s = collect_scores(model, ParquetBatchStream(validated['base'] / f"{validated['target']}_val", 1024, False, seed, True))
        runs[name] = {'target_development': compute_metrics(y, s, threshold),
                      'threshold': threshold, 'score_analysis': score_diagnostics(y, s, threshold),
                      'selected_stage': result['selected_stage'] if result else 'source_only'}
        runs[name]['source_val'] = checked[name][2] if result else baseline['source_val']
        print(f"{name}: AP={runs[name]['target_development']['pr_auc']:.6f}")
    baseline_ap = runs['source_only']['target_development']['pr_auc']
    report = {**provenance(validated, checkpoint, direction, seed, config_path),
              'data_role': 'target_validation_development_only', 'runs': runs,
              'paired_delta_ap': {exp: runs[exp]['target_development']['pr_auc'] - baseline_ap for exp in EXPERIMENTS},
              'e2_minus_e1_ap': runs['E2']['target_development']['pr_auc'] - runs['E1']['target_development']['pr_auc']}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(f'Saved: {output}')


def aggregate(config_path, directions, seeds, output_name):
    from evaluation.proposal_aggregate import seed_statistics
    load_config(config_path)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Seeds must be nonempty and unique')
    output = revision_path('result_root', None) / 'extensions' / VERSION / output_name
    if output.exists():
        raise FileExistsError(f'Aggregate directory already exists: {output}')
    rows, summaries = [], []
    for direction in directions:
        for seed in seeds:
            validated, checkpoint, baseline = reference(direction, seed)
            for exp in EXPERIMENTS:
                _, result, _ = validate_run(direction, seed, exp, config_path, validated, checkpoint)
                rows.append({'direction': direction, 'seed': seed, 'experiment': exp,
                             'selected_stage': result['selected_stage'],
                             **{key: result['target_development'][key] for key in METRICS},
                             'delta_ap': result['target_development']['pr_auc'] - baseline['target_development']['pr_auc']})
        for exp in EXPERIMENTS:
            group = [row for row in rows if row['direction'] == direction and row['experiment'] == exp]
            summaries.append({'direction': direction, 'experiment': exp,
                              'metrics': {key: seed_statistics([row[key] for row in group]) for key in (*METRICS, 'delta_ap')}})
        paired = []
        for seed in seeds:
            pair = {row['experiment']: row['pr_auc'] for row in rows if row['direction'] == direction and row['seed'] == seed}
            paired.append(pair['E2'] - pair['E1'])
        summaries.append({'direction': direction, 'experiment': 'E2_minus_E1',
                          'metrics': {'delta_ap': seed_statistics(paired)}})
    output.mkdir(parents=True)
    with (output / 'summary.json').open('x', encoding='utf-8') as stream:
        json.dump({'extension_id': VERSION, 'data_role': 'target_validation_development_only',
                   'per_run': rows, 'summary': summaries}, stream, indent=2, allow_nan=False)
    with (output / 'per_run.csv').open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Saved: {output}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=['train', 'evaluate', 'aggregate'])
    parser.add_argument('--experiment', choices=EXPERIMENTS)
    parser.add_argument('--direction', choices=[*DIRECTIONS, 'both'], default='both')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46])
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--output-name', help='New evaluation/aggregate directory; existing outputs are never overwritten')
    args = parser.parse_args()
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error('Seeds must be nonempty and unique')
    name = args.output_name or ('aggregate' if args.stage == 'aggregate' else 'evaluation')
    if not name or Path(name).name != name or name in ('.', '..', *EXPERIMENTS):
        parser.error('--output-name must be a distinct directory name')
    directions = DIRECTIONS if args.direction == 'both' else (args.direction,)
    if args.stage == 'aggregate':
        aggregate(args.config, directions, args.seeds, name)
        return
    for direction in directions:
        for seed in args.seeds:
            if args.stage == 'evaluate':
                evaluate(direction, seed, args.config, name)
            else:
                for exp in (args.experiment,) if args.experiment else EXPERIMENTS:
                    train(direction, seed, exp, args.config)


if __name__ == '__main__':
    main()
