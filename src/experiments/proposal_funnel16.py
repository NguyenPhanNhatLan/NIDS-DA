"""CE / source SupCon / class-aware MK-MMD ablation on canonical inputs."""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import average_precision_score, roc_curve

from evaluation.baseline import compute_metrics, select_f1_threshold
from evaluation.proposal_score_diagnostic import score_diagnostics
from features.common_features import COMMON_FEATURES
from models.funnel16 import NIDSFunnel16
from training.baseline import set_seed
from training.data_revision import verify_revision, revision_path, require_development_open, file_sha256
from training.proposal_data import ParquetBatchStream, split_sha256
from training.proposal_mkmmd import multi_kernel_mmd_loss

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / 'configs/proposal_funnel16.json'
STAGES = ('source-only', 'source-supcon', 'ce-control', 'class-mkmmd')


def supervised_contrastive_loss(z, labels, temperature):
    """One-view SupCon, excluding self and anchors without positive partners."""
    z = F.normalize(z, dim=1)
    similarities = z @ z.T / temperature
    diagonal = torch.eye(len(z), device=z.device, dtype=torch.bool)
    positive = labels[:, None].eq(labels[None, :]) & ~diagonal
    valid = positive.sum(1) > 0
    if not valid.any():
        return z.sum() * 0.0
    similarities = similarities.masked_fill(diagonal, -torch.inf)
    log_prob = similarities - torch.logsumexp(similarities, dim=1, keepdim=True)
    positive_log_prob = log_prob.masked_fill(~positive, 0.0).sum(1)
    return -(positive_log_prob[valid] / positive.sum(1)[valid]).mean()


@torch.no_grad()
def scores(model, path, seed):
    model.eval()
    device = next(model.parameters()).device
    labels, values = [], []
    for x, y in ParquetBatchStream(path, 1024, False, seed, True):
        _, _, logits = model(x.to(device))
        labels.append(y.numpy())
        values.append(logits.softmax(1)[:, 1].cpu().numpy())
    if not labels:
        raise ValueError(f'Empty validation split: {path}')
    return np.concatenate(labels), np.concatenate(values)


def context(direction, seed, config_path):
    config = json.loads(config_path.read_text())
    if config['input_dim'] != len(COMMON_FEATURES) or config['latent_dim'] != 16:
        raise ValueError('Expected canonical five inputs and a 16D bottleneck')
    if config['missing_class_policy'] != 'skip_batch':
        raise ValueError('Both classes must be eligible, otherwise skip alignment')
    if (config['batch_size'] < 2 or config['epochs'] < 1 or config['patience'] < 1
            or config['supcon_temperature'] <= 0 or config['lambda_warmup_epochs'] < 1
            or not 0.5 <= config['pseudo_confidence'] <= 1
            or config['alpha_supcon'] < 0 or config['lambda_alignment'] < 0
            or not config['kernel_scales'] or any(s <= 0 for s in config['kernel_scales'])
            or config['pseudo_probe_rows'] < 2
            or any(not 0 < fpr < 1 for fpr in config['source_fpr_levels'])):
        raise ValueError('Invalid Funnel16 hyperparameters')
    revision = verify_revision()
    if not revision:
        raise ValueError('A frozen data revision is required')
    source, target = ('unsw', 'cicids') if direction == 'unsw_to_cicids' else ('cicids', 'unsw')
    base = revision_path('feature_root', None) / direction
    paths = {name: base / f'{domain}_{split}' for name, domain, split in (
        ('source_train', source, 'train'), ('source_val', source, 'val'),
        ('target_train', target, 'train'), ('target_val', target, 'val'))}
    provenance = {
        **revision, 'direction': direction, 'seed': seed, 'architecture': 'NIDSFunnel16',
        'features': list(COMMON_FEATURES), 'config_sha256': file_sha256(config_path),
        'prepared_split_sha256': {name: split_sha256(path) for name, path in paths.items()},
    }
    model_root = revision_path('model_root', None) / 'funnel_16d'
    result_root = revision_path('result_root', None) / 'funnel_16d'
    return config, paths, provenance, model_root, result_root


def checkpoint_path(root, stage, direction, seed):
    return root / stage / direction / f'seed{seed}.pt'


def load_checkpoint(path, provenance, stage):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    for key, value in {**provenance, 'method': stage}.items():
        if checkpoint.get(key) != value:
            raise ValueError(f'Stale checkpoint {path}: {key}')
    return checkpoint


@torch.no_grad()
def probe_predictions(model, probe, confidence):
    model.eval()
    _, _, logits = model(probe)
    probabilities, labels = logits.softmax(1).max(1)
    return labels.cpu(), (probabilities >= confidence).cpu()


def train(direction, seed, stage, config_path):
    require_development_open()
    config, paths, provenance, model_root, result_root = context(direction, seed, config_path)
    output = checkpoint_path(model_root, stage, direction, seed)
    report_path = result_root / stage / direction / f'seed{seed}.json'
    if output.exists() or report_path.exists():
        raise FileExistsError(f'Run already exists or is incomplete: {output}, {report_path}')
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NIDSFunnel16(len(COMMON_FEATURES)).to(device)
    continuation = stage in ('class-mkmmd', 'ce-control')
    parent_hash = None
    if continuation:
        parent = checkpoint_path(model_root, 'source-supcon', direction, seed)
        checkpoint = load_checkpoint(parent, provenance, 'source-supcon')
        model.load_state_dict(checkpoint['model_state_dict'])
        parent_hash = file_sha256(parent)
    counts = torch.zeros(2)
    for _, y in ParquetBatchStream(paths['source_train'], 4096, False, seed, True):
        counts += torch.bincount(y, minlength=2)
    if (counts == 0).any():
        raise ValueError('Source training must contain both classes')
    criterion = nn.CrossEntropyLoss(weight=(counts.sum() / (2 * counts)).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'],
                                 weight_decay=config['weight_decay'])
    source_loader = ParquetBatchStream(paths['source_train'], config['batch_size'], True, seed, True, drop_last=True)
    target_loader = ParquetBatchStream(paths['target_train'], config['batch_size'], True, seed, False, drop_last=True)
    probe = None
    previous = None
    if continuation:
        parts, remaining = [], config['pseudo_probe_rows']
        for x in ParquetBatchStream(paths['target_train'], 1024, False, seed, False):
            parts.append(x[:remaining])
            remaining -= len(parts[-1])
            if remaining == 0:
                break
        if not parts:
            raise ValueError('Target training is empty')
        probe = torch.cat(parts).to(device)
        previous = probe_predictions(model, probe, config['pseudo_confidence'])
    y, s = scores(model, paths['source_val'], seed)
    initial_ap = float(average_precision_score(y, s))
    if continuation and not math.isclose(initial_ap, checkpoint['best_source_val_ap'], abs_tol=1e-3):
        raise ValueError('Source SupCon AP replay differs')
    best_ap = initial_ap if continuation else -math.inf
    best_epoch, best_state = 0, deepcopy(model.state_dict())
    history = [{'epoch': 0, 'source_val_ap': initial_ap, 'stage': 'source_pretrained' if continuation else 'random_initialization'}]
    stale = 0
    alpha = config['alpha_supcon'] if stage != 'source-only' else 0.0
    for epoch in range(1, config['epochs'] + 1):
        model.train()
        if continuation:
            for module in model.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        weight = (config['lambda_alignment'] * min(epoch / config['lambda_warmup_epochs'], 1.0)
                  if stage == 'class-mkmmd' else 0.0)
        target_iterator = iter(target_loader) if continuation else None
        totals = {'ce': 0.0, 'supcon': 0.0, 'alignment': 0.0}
        steps = aligned = eligible_batches = 0
        eligibility_counts = {'none': 0, 'normal_only': 0, 'attack_only': 0, 'both': 0}
        predicted, accepted = [0, 0], [0, 0]
        for x, labels in source_loader:
            x, labels = x.to(device), labels.to(device)
            z, z_align, logits = model(x)
            ce = criterion(logits, labels)
            supcon = supervised_contrastive_loss(z, labels, config['supcon_temperature']) if alpha else z.sum() * 0
            alignment = z.sum() * 0
            if continuation:
                try:
                    tx = next(target_iterator)
                except StopIteration:
                    target_iterator = iter(target_loader)
                    try:
                        tx = next(target_iterator)
                    except StopIteration:
                        raise ValueError('Target training has no full batches') from None
                _, tz_align, target_logits = model(tx.to(device))
                confidence, pseudo = target_logits.detach().softmax(1).max(1)
                mask = confidence >= config['pseudo_confidence']
                for label in (0, 1):
                    predicted[label] += int((pseudo == label).sum())
                    accepted[label] += int((mask & (pseudo == label)).sum())
                eligible_classes = [c for c in (0, 1) if int((labels == c).sum()) >= 2
                                    and int((mask & (pseudo == c)).sum()) >= 2]
                eligibility_key = ('both' if len(eligible_classes) == 2 else
                                   'normal_only' if eligible_classes == [0] else
                                   'attack_only' if eligible_classes == [1] else 'none')
                eligibility_counts[eligibility_key] += 1
                eligible = len(eligible_classes) == 2
                eligible_batches += int(eligible)
                if weight > 0 and eligible:
                    losses = [multi_kernel_mmd_loss(z_align[labels == c],
                              tz_align[mask & (pseudo == c)], config['kernel_scales'])[0]
                              for c in (0, 1)]
                    alignment = torch.stack(losses).mean()
                    aligned += 1
            loss = ce + alpha * supcon + weight * alignment
            if not torch.isfinite(loss):
                raise ValueError('Non-finite training loss')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            steps += 1
            for key, value in (('ce', ce), ('supcon', supcon), ('alignment', alignment)):
                totals[key] += float(value.detach())
        if not steps:
            raise ValueError('Source training has no full batches')
        y, s = scores(model, paths['source_val'], seed)
        ap = float(average_precision_score(y, s))
        row = {'epoch': epoch, 'source_val_ap': ap, 'lambda_alignment': weight,
               **{key: value / steps for key, value in totals.items()},
               'aligned_batches': aligned, 'skipped_alignment_batches': steps - aligned,
               'alignment_enabled': stage == 'class-mkmmd',
               'both_classes_eligible_batches': eligible_batches,
               'eligible_batch_counts': eligibility_counts,
               'predicted_per_class': predicted, 'accepted_per_class': accepted,
               'acceptance_rate_per_predicted_class': [a / n if n else None for a, n in zip(accepted, predicted)]}
        if probe is not None:
            current = probe_predictions(model, probe, config['pseudo_confidence'])
            both = previous[1] & current[1]
            row['fixed_target_train_probe'] = {
                'rows': len(probe), 'label_flip_rate': float((current[0] != previous[0]).float().mean()),
                'accepted_both_epochs': int(both.sum()),
                'accepted_label_flip_rate': float((current[0][both] != previous[0][both]).float().mean()) if both.any() else None,
                'target_labels_used': False,
            }
            previous = current
        history.append(row)
        improvement = ap - best_ap
        if ap > best_ap:
            best_ap, best_epoch, best_state = ap, epoch, deepcopy(model.state_dict())
        stale = 0 if improvement > config['min_delta'] else stale + 1
        print(f'{stage} epoch {epoch:03d}: source AP={ap:.6f}, lambda={weight:.5f}, aligned={aligned}/{steps}')
        if stale >= config['patience']:
            break
    model.load_state_dict(best_state)
    y, s = scores(model, paths['source_val'], seed)
    threshold = select_f1_threshold(y, s)
    metadata = {**provenance, 'method': stage, 'config': config,
                'parent_checkpoint_sha256': parent_hash, 'best_epoch': best_epoch,
                'best_source_val_ap': best_ap, 'threshold': threshold,
                'selected_stage': 'source_pretrained' if continuation and best_epoch == 0 else 'trained',
                'threshold_selection': 'source validation F1',
                'checkpoint_selection': 'source validation AP', 'target_labels_used_training': False,
                'target_labels_used_selection': False, 'history': history}
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({**metadata, 'model_state_dict': model.state_dict()}, output)
    with report_path.open('x', encoding='utf-8') as stream:
        json.dump({**metadata, 'checkpoint': str(output),
                   'source_val': compute_metrics(y, s, threshold)}, stream, indent=2, allow_nan=False)
    print(f'Saved: {report_path}')


def threshold_at_source_fpr(labels, values, limit):
    # ROC thresholds use score >= threshold, matching compute_metrics.
    fpr, _, thresholds = roc_curve(labels, values, drop_intermediate=False)
    threshold = thresholds[np.flatnonzero(fpr <= limit)[-1]]
    return float(threshold) if np.isfinite(threshold) else float(np.nextafter(values.max(), np.inf))


def evaluate(direction, seed, config_path, evaluation_name):
    config, paths, provenance, model_root, result_root = context(direction, seed, config_path)
    output = result_root / evaluation_name / direction / f'seed{seed}.json'
    if output.exists():
        raise FileExistsError(f'Evaluation already exists: {output}; choose a new --evaluation-name')
    checkpoints = {stage: load_checkpoint(checkpoint_path(model_root, stage, direction, seed), provenance, stage)
                   for stage in STAGES}
    parent_hash = file_sha256(checkpoint_path(model_root, 'source-supcon', direction, seed))
    if any(checkpoints[stage]['parent_checkpoint_sha256'] != parent_hash
           for stage in ('class-mkmmd', 'ce-control')):
        raise ValueError('Continuation runs used different source SupCon checkpoints')
    runs, validated = {}, {}
    # Replay all source selections before reading target validation labels.
    for stage, checkpoint in checkpoints.items():
        model = NIDSFunnel16(len(COMMON_FEATURES))
        model.load_state_dict(checkpoint['model_state_dict'])
        y, s = scores(model, paths['source_val'], seed)
        if (not math.isclose(float(average_precision_score(y, s)), checkpoint['best_source_val_ap'], abs_tol=1e-3)
                or not math.isclose(select_f1_threshold(y, s), checkpoint['threshold'], abs_tol=1e-6)):
            raise ValueError(f'Source validation replay differs: {stage}')
        validated[stage] = (model, y, s)
    for stage, (model, y, s) in validated.items():
        checkpoint = checkpoints[stage]
        ty, ts = scores(model, paths['target_val'], seed)
        threshold = checkpoint['threshold']
        fixed_fpr = {}
        for limit in config['source_fpr_levels']:
            selected = threshold_at_source_fpr(y, s, limit)
            # Labeled target ROC is explicitly diagnostic, never used for deployment.
            fpr, recall, _ = roc_curve(ty, ts, drop_intermediate=False)
            fixed_fpr[str(limit)] = {
                'threshold_from_source': selected,
                'source_val': compute_metrics(y, s, selected),
                'target_val': compute_metrics(ty, ts, selected),
                'target_oracle_recall_at_fpr_diagnostic_only': float(recall[fpr <= limit].max()),
            }
        runs[stage] = {'selected_stage': checkpoint['selected_stage'], 'threshold': threshold,
                       'checkpoint_sha256': file_sha256(checkpoint_path(model_root, stage, direction, seed)),
                       'source_val': compute_metrics(y, s, threshold),
                       'target_development': compute_metrics(ty, ts, threshold),
                       'target_score_analysis': score_diagnostics(ty, ts, threshold),
                       'source_fpr_operating_points': fixed_fpr}
        print(f"{stage}: target AP={runs[stage]['target_development']['pr_auc']:.6f}")
    ap = {stage: run['target_development']['pr_auc'] for stage, run in runs.items()}
    result = {**provenance, 'data_role': 'target_validation_development_only', 'runs': runs,
              'paired_delta_ap': {
                  'supcon_minus_ce': ap['source-supcon'] - ap['source-only'],
                  'continued_control_minus_supcon': ap['ce-control'] - ap['source-supcon'],
                  'alignment_minus_supcon': ap['class-mkmmd'] - ap['source-supcon'],
                  'alignment_minus_continued_control': ap['class-mkmmd'] - ap['ce-control']}}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(f'Saved: {output}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--direction', required=True, choices=['unsw_to_cicids', 'cicids_to_unsw'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--stage', required=True, choices=[*STAGES, 'evaluate'])
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--evaluation-name', default='evaluation')
    args = parser.parse_args()
    if Path(args.evaluation_name).name != args.evaluation_name or args.evaluation_name in ('.', '..', *STAGES):
        parser.error('--evaluation-name must be a distinct directory name')
    if args.stage == 'evaluate':
        evaluate(args.direction, args.seed, args.config, args.evaluation_name)
    else:
        train(args.direction, args.seed, args.stage, args.config)


if __name__ == '__main__':
    main()
