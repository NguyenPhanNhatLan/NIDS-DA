"""Training helpers for projected SupCon; original proposal trainers are unchanged."""
from copy import deepcopy
import math

import numpy as np
import torch
from torch import nn

from evaluation.baseline import collect_scores, compute_metrics, select_f1_threshold
from models.baseline import BaselineMLP
from models.contrastive_projection import ContrastiveProjection
from training.baseline import set_seed
from training.proposal_mkmmd import multi_kernel_mmd_loss
from training.supcon import supervised_contrastive_loss


def initialize(source_state, input_dim, seed, device):
    set_seed(seed)
    model = BaselineMLP(input_dim)
    model.load_state_dict(source_state, strict=True)
    projection = ContrastiveProjection()
    return model.to(device), projection.to(device)


def effective_lambda(config, epoch):
    warmup = config['mmd_warmup_epochs']
    return config['lambda_mmd'] * (min(1.0, epoch / warmup) if warmup else 1.0)


def frozen_bn_training(model):
    model.train()
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def losses(model, projection, source_x, source_y, target_x, criterion, config, epoch):
    source_z, logits = model(source_x)
    ce = criterion(logits, source_y)
    if config['alpha_supcon']:
        contrastive, stats = supervised_contrastive_loss(projection(source_z), source_y, config['temperature'])
    else:
        contrastive = source_z.sum() * 0
        stats = {'class_counts': torch.bincount(source_y, minlength=2).tolist(),
                 'valid_anchors': 0, 'valid_anchors_per_class': [0, 0],
                 'skipped': True, 'skip_reason': 'disabled'}
    weight = effective_lambda(config, epoch)
    mmd, bandwidth = source_z.new_zeros(()), source_z.new_zeros(())
    if weight:
        if not isinstance(target_x, torch.Tensor):
            raise ValueError('Target loader must return a feature tensor only')
        if len(target_x) != len(source_x):
            raise ValueError('E2 requires an equal-sized unlabeled target feature batch')
        target_z, _ = model(target_x)
        mmd, bandwidth = multi_kernel_mmd_loss(source_z, target_z, config['kernel_scales'])
    total = ce + config['alpha_supcon'] * contrastive + weight * mmd
    if not torch.isfinite(total):
        raise ValueError('Non-finite adaptation loss')
    return total, {'ce': ce, 'supcon': contrastive, 'mkmmd': mmd, 'bandwidth': bandwidth}, stats


def source_evaluation(model, loader):
    y, scores = collect_scores(model, loader)
    if len(np.unique(y)) != 2:
        raise ValueError('Source validation must contain both classes')
    threshold = select_f1_threshold(y, scores)
    return compute_metrics(y, scores, threshold)


def train(model, projection, source, target, validation, counts, config, expected_ap):
    device = next(model.parameters()).device
    counts = torch.as_tensor(counts, dtype=torch.float32, device=device)
    if counts.shape != (2,) or (counts <= 0).any():
        raise ValueError('Both source classes are required')
    criterion = nn.CrossEntropyLoss(weight=counts.sum() / (2 * counts))
    settings = config['training']
    optimizer = torch.optim.Adam([*model.parameters(), *projection.parameters()],
                                 lr=settings['learning_rate'], weight_decay=settings['weight_decay'])
    metrics = source_evaluation(model, validation)
    if not math.isclose(metrics['pr_auc'], expected_ap, abs_tol=1e-3, rel_tol=0):
        raise ValueError('Pretrained source AP replay differs')
    best_ap, best_epoch = metrics['pr_auc'], 0
    best_model, best_projection = deepcopy(model.state_dict()), deepcopy(projection.state_dict())
    history = [{'epoch': 0, 'stage': 'source_pretrained', 'source_val_ap': best_ap,
                'source_val_roc_auc': metrics['roc_auc']}]
    stale = 0
    for epoch in range(1, settings['epochs'] + 1):
        frozen_bn_training(model)
        projection.train()
        target_iter = iter(target) if config['lambda_mmd'] and target is not None else None
        totals = {key: 0.0 for key in ('total_loss', 'ce', 'supcon', 'mkmmd', 'bandwidth')}
        batch_records = []
        for x, y in source:
            tx = None
            if config['lambda_mmd']:
                try:
                    tx = next(target_iter)
                except StopIteration:
                    target_iter = iter(target)
                    try:
                        tx = next(target_iter)
                    except StopIteration:
                        raise ValueError('Target training has no full batches') from None
                if not isinstance(tx, torch.Tensor):
                    raise ValueError('Target training must return features only')
                tx = tx.to(device)
            optimizer.zero_grad()
            total, terms, stats = losses(model, projection, x.to(device), y.to(device), tx, criterion, config, epoch)
            total.backward()
            optimizer.step()
            totals['total_loss'] += float(total.detach())
            for key, value in terms.items():
                totals[key] += float(value.detach())
            batch_records.append({'batch': len(batch_records), **stats})
        steps = len(batch_records)
        if not steps:
            raise ValueError('Source training has no full batches')
        metrics = source_evaluation(model, validation)
        row = {'epoch': epoch, 'source_val_ap': metrics['pr_auc'], 'source_val_roc_auc': metrics['roc_auc'],
               'effective_lambda': effective_lambda(config, epoch),
               **{key: value / steps for key, value in totals.items()},
               'valid_contrastive_anchors': sum(row['valid_anchors'] for row in batch_records),
               'skipped_contrastive_batches': sum(row['skipped'] for row in batch_records),
               'batch_diagnostics': batch_records}
        history.append(row)
        print(f"{config['experiment']} epoch {epoch:03d}: AP={metrics['pr_auc']:.6f}, ROC={metrics['roc_auc']:.6f}, lambda={row['effective_lambda']:.5f}, skipped SupCon={row['skipped_contrastive_batches']}/{steps}")
        improvement = metrics['pr_auc'] - best_ap
        if metrics['pr_auc'] > best_ap:
            best_ap, best_epoch = metrics['pr_auc'], epoch
            best_model, best_projection = deepcopy(model.state_dict()), deepcopy(projection.state_dict())
        stale = 0 if improvement > settings['min_delta'] else stale + 1
        if stale >= settings['patience']:
            break
    model.load_state_dict(best_model)
    projection.load_state_dict(best_projection)
    model.eval()
    return best_epoch, best_ap, history
