"""Training helpers shared by source-only, adaptation, and diagnostics."""

import random

import numpy as np
import torch
from torch import nn


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def balanced_cross_entropy(counts, device):
    """Weight Normal and Attack inversely to their source-training counts."""
    counts = torch.as_tensor(counts, dtype=torch.float32)
    if (counts == 0).any():
        raise ValueError("Training data must contain both classes.")
    weights = (counts.sum() / (2.0 * counts)).to(device)
    return nn.CrossEntropyLoss(weight=weights)


def freeze_bn_stats(model):
    """Keep source BatchNorm statistics fixed while adapting model weights."""
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def weighted_gradient_norm(term, parameters):
    """Measure a weighted loss gradient without changing accumulated gradients."""
    if not term.requires_grad:
        return 0.0
    gradients = torch.autograd.grad(term, parameters, retain_graph=True, allow_unused=True)
    norms = [gradient.detach().norm() for gradient in gradients if gradient is not None]
    return float(torch.stack(norms).norm()) if norms else 0.0
