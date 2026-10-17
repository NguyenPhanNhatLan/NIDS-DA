"""Single-RBF MMD with a detached batch median bandwidth."""

import torch


def estimate_bandwidth_squared(source_features, target_features):
    with torch.no_grad():
        combined = torch.cat([source_features.detach(), target_features.detach()], dim=0)
        distance_squared = torch.cdist(combined, combined).square()
        positive = distance_squared[distance_squared > 1e-12]
        if positive.numel() == 0:
            return torch.tensor(1.0, device=combined.device)
        return torch.median(positive).clamp_min(1e-6)


def rbf_kernel(x, y, bandwidth_squared):
    distance_squared = torch.cdist(x, y).square()
    return torch.exp(-distance_squared / (2.0 * bandwidth_squared))


def mmd_loss(source_features, target_features):
    """Biased empirical MMD², using equal-sized prefixes of the two batches."""
    batch_size = min(len(source_features), len(target_features))
    source_features = source_features[:batch_size]
    target_features = target_features[:batch_size]
    bandwidth_squared = estimate_bandwidth_squared(source_features, target_features)
    source_kernel = rbf_kernel(source_features, source_features, bandwidth_squared)
    target_kernel = rbf_kernel(target_features, target_features, bandwidth_squared)
    cross_kernel = rbf_kernel(source_features, target_features, bandwidth_squared)
    loss = source_kernel.mean() + target_kernel.mean() - 2.0 * cross_kernel.mean()
    return loss, torch.sqrt(bandwidth_squared)
