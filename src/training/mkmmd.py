"""Biased MK-MMD with an equally weighted mixture of RBF kernels.

Scales multiply sigma squared (not sigma). The default mixture averages five
kernels at [0.25, 0.5, 1, 2, 4] times the detached median bandwidth squared.
Batch truncation and median heuristic match the frozen single-RBF implementation.
"""
import math

import torch

from training.adaptation import estimate_bandwidth_squared


def mk_mmd_loss(source_features, target_features,
                scales=(0.25, 0.5, 1.0, 2.0, 4.0), bandwidth_squared=None):
    """Return (biased MMD squared, base sigma), compatible with mmd_loss.

    Optional fixed bandwidth is useful for reproducible diagnostics/gradcheck.
    Bandwidth is always detached; gradients flow through all kernel distances.
    Diagonals are included. Tiny negative rounding errors are not clamped.
    """
    s, t = source_features, target_features
    if (s.ndim != 2 or t.ndim != 2 or s.shape[1] != t.shape[1]
            or min(len(s), len(t), s.shape[1]) == 0):
        raise ValueError("Expected nonempty [N,D] and [M,D] tensors with matching D")
    if s.device != t.device or s.dtype != t.dtype or s.dtype not in (torch.float32, torch.float64):
        raise ValueError("Inputs must share device and float32/float64 dtype")
    if not torch.isfinite(s).all() or not torch.isfinite(t).all():
        raise ValueError("MK-MMD inputs must be finite")
    scales = tuple(float(value) for value in scales)
    if not scales or any(not math.isfinite(value) or value <= 0 for value in scales):
        raise ValueError("Kernel scales must be positive finite values")
    n = min(len(s), len(t))
    s, t = s[:n], t[:n]
    base = (estimate_bandwidth_squared(s, t) if bandwidth_squared is None else
            torch.as_tensor(bandwidth_squared, dtype=s.dtype, device=s.device))
    base = base.detach().to(dtype=s.dtype, device=s.device)
    if base.numel() != 1 or not torch.isfinite(base).all() or (base <= 0).any():
        raise ValueError("bandwidth_squared must be a positive finite scalar")
    base = base.reshape(())
    distances = (torch.cdist(s, s).square(), torch.cdist(t, t).square(), torch.cdist(s, t).square())
    terms = []
    for scale in scales:
        denominator = 2.0 * base * scale
        if not torch.isfinite(denominator) or denominator <= 0:
            raise ValueError("Scaled bandwidth overflow/underflow")
        ss, tt, st = [torch.exp(-d / denominator).mean() for d in distances]
        terms.append(ss + tt - 2.0 * st)
    loss = torch.stack(terms).mean()
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite MK-MMD loss")
    return loss, base.sqrt()
