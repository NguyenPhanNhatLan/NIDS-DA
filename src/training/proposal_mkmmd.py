"""Three-scale marginal MK-MMD for the proposal_v2 shared latent space."""
import argparse
from pathlib import Path

import torch

from training.adaptation import estimate_bandwidth_squared

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/proposal_mkmmd_v2.json"


def multi_kernel_mmd_loss(source_z, target_z, scales):
    if not scales or any(scale <= 0 for scale in scales):
        raise ValueError("MK-MMD bandwidth scales must be positive")
    n = min(len(source_z), len(target_z))
    if n < 2:
        raise ValueError("MK-MMD needs at least two rows per domain")
    source_z, target_z = source_z[:n], target_z[:n]
    base_bandwidth_squared = estimate_bandwidth_squared(source_z, target_z)
    d_ss = torch.cdist(source_z, source_z).square()
    d_tt = torch.cdist(target_z, target_z).square()
    d_st = torch.cdist(source_z, target_z).square()
    losses = []
    for scale in scales:
        bandwidth_squared = base_bandwidth_squared * float(scale) ** 2
        k_ss = torch.exp(-d_ss / (2 * bandwidth_squared))
        k_tt = torch.exp(-d_tt / (2 * bandwidth_squared))
        k_st = torch.exp(-d_st / (2 * bandwidth_squared))
        losses.append(k_ss.mean() + k_tt.mean() - 2 * k_st.mean())
    return torch.stack(losses).mean(), torch.sqrt(base_bandwidth_squared)


def main():
    from training.proposal_mmd import run

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direction", required=True, choices=["unsw_to_cicids", "cicids_to_unsw"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.direction, args.seed, DEFAULT_CONFIG)


if __name__ == "__main__":
    main()
