from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from evaluation.baseline import (
    collect_scores,
    compute_metrics,
    select_f1_threshold,
)
from features.common_features import COMMON_FEATURES
from models.baseline import BaselineMLP
from training.baseline import (
    make_loader,
    set_seed,
    train_baseline,
)

ROOT = Path(__file__).resolve().parents[2]

FEATURE_ROOT = ROOT / "data" / "features" / "proposal_v1"

RESULT_ROOT = ROOT / "results" / "proposal_v1" / "source_only"


def direction_domains(direction):
    if direction == "unsw_to_cicids":
        return "unsw", "cicids"

    if direction == "cicids_to_unsw":
        return "cicids", "unsw"

    raise ValueError(f"Unknown direction: {direction}")


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def count_classes(path, input_dim):
    loader = make_loader(
        path,
        input_dim=input_dim,
        batch_size=4096,
        training=False,
    )

    counts = torch.zeros(
        2,
        dtype=torch.long,
    )

    for _, labels in loader:
        counts += torch.bincount(
            labels,
            minlength=2,
        )

    if (counts == 0).any():
        raise ValueError(f"Both classes required: " f"{counts.tolist()}")

    return counts.tolist()


def evaluate_split(
    model,
    path,
    input_dim,
    threshold,
):
    loader = make_loader(
        path,
        input_dim=input_dim,
        batch_size=1024,
        training=False,
    )

    labels, scores = collect_scores(
        model,
        loader,
    )

    return compute_metrics(
        labels,
        scores,
        threshold,
    )


def run(direction, seed=42):
    source, target = direction_domains(direction)

    input_dim = len(COMMON_FEATURES)

    if input_dim != 10:
        raise ValueError(f"Expected 10 common features, " f"got {input_dim}")

    base = FEATURE_ROOT / direction

    source_train_path = base / f"{source}_train"

    source_val_path = base / f"{source}_val"

    source_test_path = base / f"{source}_test"

    target_test_path = base / f"{target}_test"


    set_seed(seed)

    device = get_device()

    print(f"\nDirection: {direction}")
    print(f"Source: {source}")
    print(f"Target: {target}")
    print(f"Seed: {seed}")
    print(f"Device: {device}")
    print(f"Input dim: {input_dim}")

    # ----------------------------------
    # Source training class counts
    # ----------------------------------

    counts = count_classes(
        source_train_path,
        input_dim,
    )

    print(f"Source class counts: {counts}")

    # ----------------------------------
    # Loaders
    # ----------------------------------

    train_loader = make_loader(
        source_train_path,
        input_dim=input_dim,
        batch_size=256,
        training=True,
    )

    val_loader = make_loader(
        source_val_path,
        input_dim=input_dim,
        batch_size=1024,
        training=False,
    )

    # ----------------------------------
    # Train ONE source model
    # ----------------------------------

    model = BaselineMLP(input_dim=input_dim).to(device)

    (
        model,
        best_epoch,
        best_val_ap,
    ) = train_baseline(
        model,
        counts,
        train_loader,
        val_loader,
        epochs=50,
        lr=0.001,
        patience=7,
    )

    # ----------------------------------
    # Threshold from SOURCE VALIDATION
    # only
    # ----------------------------------

    threshold_loader = make_loader(
        source_val_path,
        input_dim=input_dim,
        batch_size=1024,
        training=False,
    )

    val_labels, val_scores = collect_scores(
        model,
        threshold_loader,
    )

    threshold = select_f1_threshold(
        val_labels,
        val_scores,
    )

    print(f"Frozen source threshold: " f"{threshold:.8f}")

    # ----------------------------------
    # Within-domain
    #
    # UNSW -> UNSW
    # or
    # CICIDS -> CICIDS
    # ----------------------------------

    within_metrics = evaluate_split(
        model,
        source_test_path,
        input_dim,
        threshold,
    )

    # ----------------------------------
    # Cross-domain
    #
    # UNSW -> CICIDS
    # or
    # CICIDS -> UNSW
    #
    # SAME model, SAME threshold.
    # ----------------------------------

    cross_metrics = evaluate_split(
        model,
        target_test_path,
        input_dim,
        threshold,
    )

    # ----------------------------------
    # Generalization degradation
    # ----------------------------------

    metric_names = [
        "pr_auc",
        "roc_auc",
        "macro_f1",
        "recall",
        "fpr",
    ]

    delta = {
        name: (cross_metrics[name] - within_metrics[name]) for name in metric_names
    }

    result = {
        "protocol": "proposal_suite_v1",
        "method": "source_only",
        "direction": direction,
        "seed": seed,
        "source_domain": source,
        "target_domain": target,
        "feature_count": input_dim,
        "features": list(COMMON_FEATURES),
        "preprocessing": (
            "source-train fit: " "median -> signed_log1p " "-> RobustScaler"
        ),
        "model": "BaselineMLP",
        "selection": {
            "checkpoint": "source validation AP",
            "threshold": "source validation F1",
            "target_labels_used": False,
        },
        "best_epoch": best_epoch,
        "best_source_val_ap": best_val_ap,
        "threshold": threshold,
        "within_domain": {
            "train": source,
            "test": source,
            **within_metrics,
        },
        "cross_domain": {
            "train": source,
            "test": target,
            **cross_metrics,
        },
        "cross_minus_within": delta,
    }

    # ----------------------------------
    # Save
    # ----------------------------------

    output_dir = RESULT_ROOT / direction

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = output_dir / f"seed{seed}.json"

    if output_path.exists():
        raise FileExistsError(f"Result already exists: " f"{output_path}")

    with output_path.open(
        "x",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            indent=2,
            allow_nan=False,
        )
        file.write("\n")

    # ----------------------------------
    # Console table
    # ----------------------------------

    print("\n" "Evaluation | AP | ROC-AUC | " "Macro-F1 | Recall | FPR")

    print(
        f"{source.upper()}->{source.upper()} | "
        f"{within_metrics['pr_auc']:.4f} | "
        f"{within_metrics['roc_auc']:.4f} | "
        f"{within_metrics['macro_f1']:.4f} | "
        f"{within_metrics['recall']:.4f} | "
        f"{within_metrics['fpr']:.4f}"
    )

    print(
        f"{source.upper()}->{target.upper()} | "
        f"{cross_metrics['pr_auc']:.4f} | "
        f"{cross_metrics['roc_auc']:.4f} | "
        f"{cross_metrics['macro_f1']:.4f} | "
        f"{cross_metrics['recall']:.4f} | "
        f"{cross_metrics['fpr']:.4f}"
    )

    print("\nCross - within:")

    for key, value in delta.items():
        print(f"{key}: {value:+.6f}")

    print(f"\nSaved: {output_path}")

    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--direction",
        required=True,
        choices=[
            "unsw_to_cicids",
            "cicids_to_unsw",
        ],
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    run(
        args.direction,
        args.seed,
    )


if __name__ == "__main__":
    main()
