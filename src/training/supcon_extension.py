"""Source-only, one-view supervised contrastive loss for the versioned extension."""
import math

import torch
from torch.nn import functional as F


def supervised_contrastive_loss(features, labels, temperature=0.1):
    """Cosine SupCon without self-pairs; omit anchors without a positive partner.

    No augmentation, projection head, target labels or pseudo-labels are used.
    Returns (loss, number_of_valid_anchors) for training diagnostics.
    """
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('SupCon temperature must be finite and positive')
    if features.ndim != 2 or labels.ndim != 1 or len(features) != len(labels):
        raise ValueError('Expected features [N,D] and labels [N]')
    if not torch.isfinite(features).all() or not ((labels == 0) | (labels == 1)).all():
        raise ValueError('Expected finite features and binary source labels')
    if len(features) < 2:
        return features.sum() * 0.0, 0
    z = F.normalize(features, dim=1)
    diagonal = torch.eye(len(z), device=z.device, dtype=torch.bool)
    positives = labels[:, None].eq(labels[None, :]) & ~diagonal
    valid = positives.sum(dim=1) > 0
    anchors = int(valid.sum())
    if not anchors:
        return features.sum() * 0.0, 0
    logits = (z @ z.T / temperature).masked_fill(diagonal, -torch.inf)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    positive_log_prob = log_prob.masked_fill(~positives, 0.0).sum(dim=1)
    return -(positive_log_prob[valid] / positives.sum(dim=1)[valid]).mean(), anchors
