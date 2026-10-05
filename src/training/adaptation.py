"""Shared single-RBF MMD kernels for proposal_v2 (no training entry point)."""
import torch

def estimate_bandwidth_squared(
    source_features,
    target_features,
):
    """
    Median heuristic.

    Dùng cả source và target representation.
    Không dùng label.
    """

    with torch.no_grad():

        combined = torch.cat(
            [
                source_features.detach(),
                target_features.detach(),
            ],
            dim=0,
        )

        distance_squared = torch.cdist(
            combined,
            combined,
        ).square()

        positive = distance_squared[
            distance_squared > 1e-12
        ]

        if positive.numel() == 0:
            return torch.tensor(
                1.0,
                device=combined.device,
            )

        bandwidth_squared = torch.median(
            positive
        )

        return bandwidth_squared.clamp_min(
            1e-6
        )

def rbf_kernel(
    x,
    y,
    bandwidth_squared,
):
    distance_squared = torch.cdist(
        x,
        y,
    ).square()

    return torch.exp(
        -distance_squared
        /
        (2.0 * bandwidth_squared)
    )

def mmd_loss(
    source_features,
    target_features,
):
    """
    Biased empirical MMD^2.

    MMD^2 =
        E[k(xs, xs')]
        + E[k(xt, xt')]
        - 2 E[k(xs, xt)]
    """

    batch_size = min(
        len(source_features),
        len(target_features),
    )

    source_features = (
        source_features[:batch_size]
    )

    target_features = (
        target_features[:batch_size]
    )

    bandwidth_squared = (
        estimate_bandwidth_squared(
            source_features,
            target_features,
        )
    )

    source_kernel = rbf_kernel(
        source_features,
        source_features,
        bandwidth_squared,
    )

    target_kernel = rbf_kernel(
        target_features,
        target_features,
        bandwidth_squared,
    )

    cross_kernel = rbf_kernel(
        source_features,
        target_features,
        bandwidth_squared,
    )

    loss = (
        source_kernel.mean()
        + target_kernel.mean()
        - 2.0 * cross_kernel.mean()
    )

    return (
        loss,
        torch.sqrt(
            bandwidth_squared
        ),
    )
