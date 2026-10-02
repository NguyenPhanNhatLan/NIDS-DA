"""Source-only MLP baseline on the fixed 10-feature proposal_v1 data."""

import argparse
import json
from pathlib import Path

import torch

from evaluation.baseline import collect_scores, compute_metrics, select_f1_threshold
from models.baseline import BaselineMLP
from training.baseline import make_loader, set_seed, train_baseline

ROOT = Path(__file__).resolve().parents[2]
INPUT_DIM = 10


def domains(direction):
    if direction == "unsw_to_cicids":
        return "unsw", "cicids"
    if direction == "cicids_to_unsw":
        return "cicids", "unsw"
    raise ValueError(f"Unknown direction: {direction}")


def count_classes(loader):
    counts = torch.zeros(2, dtype=torch.long)
    for _, labels in loader:
        counts += torch.bincount(labels, minlength=2)
    return counts.tolist()


def run(direction, seed=42):
    source, target = domains(direction)
    base = ROOT / "data/features/proposal_v1" / direction
    set_seed(seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    counts = count_classes(
        make_loader(
            base / f"{source}_train", INPUT_DIM, batch_size=4096, training=False
        )
    )
    source_train = make_loader(
        base / f"{source}_train", INPUT_DIM, batch_size=256, training=True
    )
    source_val = make_loader(
        base / f"{source}_val", INPUT_DIM, batch_size=256, training=False
    )
    target_test = make_loader(
        base / f"{target}_test", INPUT_DIM, batch_size=256, training=False
    )

    model = BaselineMLP(INPUT_DIM).to(device)
    model, best_epoch, best_ap = train_baseline(
        model, counts, source_train, source_val, epochs=50, lr=1e-3, patience=7
    )

    source_labels, source_scores = collect_scores(model, source_val)
    threshold = select_f1_threshold(source_labels, source_scores)
    source_metrics = compute_metrics(source_labels, source_scores, threshold)
    target_labels, target_scores = collect_scores(model, target_test)
    target_metrics = compute_metrics(target_labels, target_scores, threshold)

    return {
        "direction": direction,
        "seed": seed,
        "input_dim": INPUT_DIM,
        "source_train_counts": counts,
        "best_epoch": best_epoch,
        "best_source_val_ap": float(best_ap),
        "threshold_from_source_val": threshold,
        "source_val": source_metrics,
        "target_test": target_metrics,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--direction", required=True, choices=["unsw_to_cicids", "cicids_to_unsw"]
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(run(args.direction, args.seed), indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
