import torch


def estimate_bandwidth_squared(source_features, target_features):
    with torch.no_grad():
        combined = torch.cat([source_features.detach(), target_features.detach()], dim=0)
        distance_squared = torch.cdist(combined, combined).square()
        positive = distance_squared[distance_squared > 1e-12]
        if positive.numel() == 0:
            return torch.tensor(1.0, device=combined.device)
        return torch.median(positive).clamp_min(1e-6)


def rbf_from_squared_distance(distance_squared, bandwidth_squared):
    return torch.exp(-distance_squared / (2.0 * bandwidth_squared))


def rbf_kernel(x, y, bandwidth_squared):
    return rbf_from_squared_distance(torch.cdist(x, y).square(), bandwidth_squared)


def kernel_mmd_loss(source_features, target_features, scales):
    batch_size = min(len(source_features), len(target_features))
    source_features = source_features[:batch_size]
    target_features = target_features[:batch_size]
    bandwidth_squared = estimate_bandwidth_squared(source_features, target_features)
    source_distances = torch.cdist(source_features, source_features).square()
    target_distances = torch.cdist(target_features, target_features).square()
    cross_distances = torch.cdist(source_features, target_features).square()
    losses = []
    for scale in scales:
        scaled_bandwidth = bandwidth_squared * float(scale) ** 2
        source_kernel = rbf_from_squared_distance(source_distances, scaled_bandwidth)
        target_kernel = rbf_from_squared_distance(target_distances, scaled_bandwidth)
        cross_kernel = rbf_from_squared_distance(cross_distances, scaled_bandwidth)
        losses.append(source_kernel.mean() + target_kernel.mean() - 2.0 * cross_kernel.mean())
    return torch.stack(losses).mean(), torch.sqrt(bandwidth_squared)


def mmd_loss(source_features, target_features):
    return kernel_mmd_loss(source_features, target_features, (1.0,))
