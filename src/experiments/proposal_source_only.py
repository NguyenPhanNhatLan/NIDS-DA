"""Source-only MLP baseline on the fixed 10-feature proposal_v1 data."""

import argparse
import json
from pathlib import Path

import torch

from evaluation.baseline import collect_scores, compute_metrics, select_f1_threshold
from features.common_features import COMMON_FEATURES
from models.baseline import BaselineMLP
from training.baseline import make_loader, set_seed, train_baseline

ROOT = Path(__file__).resolve().parents[2]
INPUT_DIM = 10
CHECKPOINT_ROOT = ROOT / "models/proposal_v1/source_only_pretrained"
RESULT_ROOT = ROOT / "results/proposal_v1/source_only_pretrained"


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
    checkpoint_path = CHECKPOINT_ROOT / direction / f"seed{seed}.pt"
    result_path = RESULT_ROOT / direction / f"seed{seed}.json"
    for path in (checkpoint_path, result_path):
        if path.exists():
            raise FileExistsError(f"Output already exists: {path}")
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
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "protocol": "proposal_source_only_pretrained_v1",
        "direction": direction,
        "seed": seed,
        "input_dim": INPUT_DIM,
        "features": list(COMMON_FEATURES),
        "best_epoch": best_epoch,
        "best_source_val_ap": float(best_ap),
        "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
    }, checkpoint_path)
    target_labels, target_scores = collect_scores(model, target_test)
    target_metrics = compute_metrics(target_labels, target_scores, threshold)

    result = {
        "direction": direction,
        "seed": seed,
        "input_dim": INPUT_DIM,
        "feature_count": INPUT_DIM,
        "features": list(COMMON_FEATURES),
        "source_train_counts": counts,
        "best_epoch": best_epoch,
        "best_source_val_ap": float(best_ap),
        "threshold_from_source_val": threshold,
        "source_val": source_metrics,
        "target_test": target_metrics,
    }
    result["checkpoint"] = str(checkpoint_path)
    with result_path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Saved source checkpoint: {checkpoint_path}")
    print(f"Saved source result: {result_path}")
    return result


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
