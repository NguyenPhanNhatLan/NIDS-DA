"""Biased MK-MMD: shared pairwise distances and mean RBF mixture."""
import math

import torch
from torch import nn


def validate_scales(scales):
    values = tuple(float(v) for v in scales)
    if not values or any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError("Kernel scales must be positive finite values")
    return values


def mk_mmd_loss(source_features, target_features,
                scales=(0.25, 0.5, 1.0, 2.0, 4.0), bandwidth_squared=None,
                validate=True, _scale_tensor=None):
    """Return (biased MMD squared, base sigma). Scales multiply sigma squared.

    One cdist supplies both the detached median and differentiable kernels.
    validate=False avoids tensor-to-Python checks; callers must check total loss.
    MKMMDLoss validates/caches scales once for the training hot path.
    """
    s, t = source_features, target_features
    if (s.ndim != 2 or t.ndim != 2 or s.shape[1] != t.shape[1]
            or min(len(s), len(t), s.shape[1]) == 0):
        raise ValueError("Expected nonempty [N,D] and [M,D] tensors with matching D")
    if s.device != t.device or s.dtype != t.dtype or s.dtype not in (torch.float32, torch.float64):
        raise ValueError("Inputs must share device and float32/float64 dtype")
    if validate and not (torch.isfinite(s).all() & torch.isfinite(t).all()):
        raise ValueError("MK-MMD inputs must be finite")
    if _scale_tensor is None:
        values = validate_scales(scales)
        scale_tensor = torch.tensor(values, device=s.device, dtype=s.dtype)
    else:
        scale_tensor = _scale_tensor
    n = min(len(s), len(t))
    combined = torch.cat((s[:n], t[:n]), dim=0)
    dist2 = torch.cdist(combined, combined).square()
    if bandwidth_squared is None:
        detached = dist2.detach()
        positive = detached[detached > 1e-12]
        base = positive.median().clamp_min(1e-6) if positive.numel() else detached.new_tensor(1.)
    else:
        base = torch.as_tensor(bandwidth_squared, dtype=s.dtype, device=s.device).detach()
    if base.numel() != 1:
        raise ValueError("bandwidth_squared must be a scalar")
    base = base.reshape(())
    denominators = 2.0 * base * scale_tensor
    if validate and not (torch.isfinite(denominators).all() & (denominators > 0).all()):
        raise ValueError("Scaled bandwidth must be positive and finite")
    # One batched exp replaces 3*K launches. Memory O(K*(2N)^2).
    kernels = torch.exp(-dist2.unsqueeze(0) / denominators[:, None, None])
    per_kernel = (kernels[:, :n, :n].mean((1, 2)) + kernels[:, n:, n:].mean((1, 2))
                  - 2.0 * kernels[:, :n, n:].mean((1, 2)))
    loss = per_kernel.mean()
    if validate and not torch.isfinite(loss):
        raise ValueError("Nonfinite MK-MMD loss")
    return loss, base.sqrt()


class MKMMDLoss(nn.Module):
    """Validate scales once; move the cached tensor to the training device once."""
    def __init__(self, scales=(.25, .5, 1., 2., 4.)):
        super().__init__()
        self.scales = validate_scales(scales)
        self.register_buffer("scale_tensor", torch.tensor(self.scales, dtype=torch.float64))

    def forward(self, source, target, bandwidth_squared=None, validate=False):
        return mk_mmd_loss(source, target, bandwidth_squared=bandwidth_squared,
                           validate=validate, _scale_tensor=self.scale_tensor.to(source))
