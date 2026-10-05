"""Biased weighted MK-MMD with fixed/unweighted source and weighted target support."""
import math

import torch
from torch import nn

from training.mkmmd import validate_scales


def _normalized_weights(weights, count, reference, name):
    if weights is None:
        return torch.full((count,), 1.0 / count, dtype=reference.dtype, device=reference.device)
    w = torch.as_tensor(weights, dtype=reference.dtype, device=reference.device)
    if w.ndim != 1 or len(w) != count:
        raise ValueError(f"{name} weights must be a vector of length {count}")
    if not torch.isfinite(w).all() or (w < 0).any():
        raise ValueError(f"{name} weights must be finite and nonnegative")
    total = w.sum()
    if not torch.isfinite(total) or total <= 0:
        raise ValueError(f"{name} weights must have positive finite mass")
    return w / total


def weighted_mk_mmd_loss(
    source_features,
    target_features,
    target_weights,
    scales=(0.25, 0.5, 1.0, 2.0, 4.0),
    bandwidth_squared=None,
    source_weights=None,
    validate=True,
    _scale_tensor=None,
):
    """Return biased weighted MK-MMD squared and base sigma.

    The kernel bandwidth follows the existing V5e/V5f convention: detached
    median squared pairwise distance over the concatenated source/target batch.
    Kernel scales multiply that base bandwidth squared.

    Source weights default to uniform. Target weights are normalized inside
    the function, so callers can provide any nonnegative relative weights.
    """
    s, t = source_features, target_features
    if (s.ndim != 2 or t.ndim != 2 or s.shape[1] != t.shape[1]
            or len(s) == 0 or len(t) == 0 or s.shape[1] == 0):
        raise ValueError("Expected nonempty [N,D] and [M,D] tensors with matching D")
    if s.device != t.device or s.dtype != t.dtype or s.dtype not in (torch.float32, torch.float64):
        raise ValueError("Inputs must share device and float32/float64 dtype")
    if validate and not (torch.isfinite(s).all() & torch.isfinite(t).all()):
        raise ValueError("Weighted MK-MMD inputs must be finite")

    if _scale_tensor is None:
        values = validate_scales(scales)
        scale_tensor = torch.tensor(values, device=s.device, dtype=s.dtype)
    else:
        scale_tensor = _scale_tensor.to(device=s.device, dtype=s.dtype)

    ws = _normalized_weights(source_weights, len(s), s, "source")
    wt = _normalized_weights(target_weights, len(t), t, "target")

    combined = torch.cat((s, t), dim=0)
    dist2 = torch.cdist(combined, combined).square()
    if bandwidth_squared is None:
        detached = dist2.detach()
        positive = detached[detached > 1e-12]
        base = positive.median().clamp_min(1e-6) if positive.numel() else detached.new_tensor(1.0)
    else:
        base = torch.as_tensor(bandwidth_squared, dtype=s.dtype, device=s.device).detach()
    if base.numel() != 1:
        raise ValueError("bandwidth_squared must be a scalar")
    base = base.reshape(())

    denominators = 2.0 * base * scale_tensor
    if validate and not (torch.isfinite(denominators).all() & (denominators > 0).all()):
        raise ValueError("Scaled bandwidth must be positive and finite")

    kernels = torch.exp(-dist2.unsqueeze(0) / denominators[:, None, None])
    ns = len(s)
    kss = kernels[:, :ns, :ns]
    ktt = kernels[:, ns:, ns:]
    kst = kernels[:, :ns, ns:]

    ss = torch.einsum("i,kij,j->k", ws, kss, ws)
    tt = torch.einsum("i,kij,j->k", wt, ktt, wt)
    st = torch.einsum("i,kij,j->k", ws, kst, wt)
    loss = (ss + tt - 2.0 * st).mean()

    if validate and not torch.isfinite(loss):
        raise ValueError("Nonfinite weighted MK-MMD loss")
    return loss, base.sqrt()


class WeightedMKMMDLoss(nn.Module):
    """Cache validated kernel scales for the weighted conditional hot path."""
    def __init__(self, scales=(0.25, 0.5, 1.0, 2.0, 4.0)):
        super().__init__()
        self.scales = validate_scales(scales)
        self.register_buffer("scale_tensor", torch.tensor(self.scales, dtype=torch.float64))

    def forward(self, source, target, target_weights, bandwidth_squared=None,
                source_weights=None, validate=False):
        return weighted_mk_mmd_loss(
            source,
            target,
            target_weights=target_weights,
            scales=self.scales,
            bandwidth_squared=bandwidth_squared,
            source_weights=source_weights,
            validate=validate,
            _scale_tensor=self.scale_tensor,
        )
