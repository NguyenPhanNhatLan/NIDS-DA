"""Class-aware MMD using source labels and detached target pseudo-labels."""
import argparse
from pathlib import Path

import torch

from training.adaptation import mmd_loss

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/proposal_class_aware_v1.json"


def class_aware_mmd_loss(source_z, source_y, target_z, target_logits, confidence=0.8):
    if not 0 <= confidence <= 1:
        raise ValueError("Pseudo-label confidence must lie in [0, 1]")
    with torch.no_grad():
        probabilities = torch.softmax(target_logits, dim=1)
        confidence_values, pseudo_labels = probabilities.max(dim=1)
        accepted = confidence_values >= confidence
    losses = []
    bandwidths = []
    for label in (0, 1):
        source_class = source_z[source_y == label]
        target_class = target_z[accepted & (pseudo_labels == label)]
        n = min(len(source_class), len(target_class))
        if n < 2:
            continue
        loss, bandwidth = mmd_loss(source_class[:n], target_class[:n])
        losses.append(loss)
        bandwidths.append(bandwidth)
    if not losses:
        zero = source_z.sum() * 0
        return zero, zero.detach()
    return torch.stack(losses).mean(), torch.stack(bandwidths).mean()


def main():
    from training.proposal_mmd import run

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direction", required=True, choices=["unsw_to_cicids", "cicids_to_unsw"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.direction, args.seed, DEFAULT_CONFIG)


if __name__ == "__main__":
    main()
