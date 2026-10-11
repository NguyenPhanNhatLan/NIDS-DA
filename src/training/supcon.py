"""Stable, class-balanced source SupCon with explicit skipped-batch diagnostics."""
import math

import torch
from torch.nn import functional as F


def supervised_contrastive_loss(projected, labels, temperature=0.1):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Temperature must be finite and positive')
    if projected.ndim != 2 or labels.ndim != 1 or len(projected) != len(labels):
        raise ValueError('Expected projected [N,D] and source labels [N]')
    if not torch.isfinite(projected).all() or not ((labels == 0) | (labels == 1)).all():
        raise ValueError('Expected finite embeddings and binary source labels')
    counts = torch.bincount(labels.long(), minlength=2).tolist()
    stats = {'class_counts': counts, 'valid_anchors': 0,
             'valid_anchors_per_class': [0, 0], 'positive_pairs': 0,
             'negative_pairs': 0, 'skipped': True, 'skip_reason': None}
    zero = projected.reshape(-1)[:1].sum() * 0.0
    if not all(counts):
        stats['skip_reason'] = 'missing_source_class'
        return zero, stats
    # Use float32 for half-precision inputs; preserve float64 for numerical checks.
    z = F.normalize(projected.float() if projected.dtype in (torch.float16, torch.bfloat16) else projected, dim=1)
    diagonal = torch.eye(len(z), dtype=torch.bool, device=z.device)
    same = labels[:, None].eq(labels[None, :])
    positive = same & ~diagonal
    negative = ~same
    valid = positive.sum(1) > 0
    stats.update(positive_pairs=int(positive.sum()), negative_pairs=int(negative.sum()),
                 valid_anchors=int(valid.sum()),
                 valid_anchors_per_class=[int((valid & (labels == c)).sum()) for c in (0, 1)])
    if not valid.any():
        stats['skip_reason'] = 'no_positive_pairs'
        return zero, stats
    logits = z @ z.T / temperature
    if not torch.isfinite(logits).all():
        raise ValueError('Temperature produces non-finite similarities')
    logits = logits.masked_fill(diagonal, -torch.inf)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    anchor_loss = -log_prob.masked_fill(~positive, 0.0).sum(1) / positive.sum(1).clamp_min(1)
    # Average anchors within each represented class, then average classes. A large
    # benign majority cannot outweigh all attack anchors simply by its count.
    class_losses = [anchor_loss[valid & (labels == c)].mean() for c in (0, 1)
                    if (valid & (labels == c)).any()]
    loss = torch.stack(class_losses).mean()
    if not torch.isfinite(loss):
        raise ValueError('Non-finite SupCon loss')
    stats.update(skipped=False, skip_reason=None)
    return loss, stats
