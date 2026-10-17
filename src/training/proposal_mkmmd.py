"""Multi-kernel MMD using the shared RBF engine."""

from training.adaptation import kernel_mmd_loss


def multi_kernel_mmd_loss(source_z, target_z, scales):
    if not scales or any(scale <= 0 for scale in scales):
        raise ValueError("MK-MMD bandwidth scales must be positive")
    if min(len(source_z), len(target_z)) < 2:
        raise ValueError("MK-MMD needs at least two rows per domain")
    return kernel_mmd_loss(source_z, target_z, scales)
