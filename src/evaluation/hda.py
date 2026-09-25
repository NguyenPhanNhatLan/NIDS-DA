import argparse
import json
from pathlib import Path

import numpy as np
import torch

from models.baseline import BaselineMLP
from models.target_encoder import (
    TargetEncoder,
    TargetModel,
)

from training.baseline import make_loader

from evaluation.baseline import (
    collect_scores,
    compute_metrics,
)


PROJECT_DIR = Path(__file__).resolve().parents[2]

FEATURE_DIR = (
    PROJECT_DIR
    / "data"
    / "features"
)

MODEL_DIR = (
    PROJECT_DIR
    / "models"
)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    seed = args.seed

    # ======================================================
    # Load source checkpoint
    # ======================================================

    source_checkpoint_path = (
        MODEL_DIR
        / "baselines"
        / f"unsw_seed{seed}.pt"
    )

    source_checkpoint = torch.load(
        source_checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    source_dim = int(
        source_checkpoint[
            "input_dim"
        ]
    )

    source_model = BaselineMLP(
        input_dim=source_dim
    )

    source_model.load_state_dict(
        source_checkpoint[
            "model_state_dict"
        ]
    )

    source_model.eval()

    # ======================================================
    # Load HDA target encoder
    # ======================================================

    hda_checkpoint_path = (
        MODEL_DIR
        / "hda"
        / (
            "unsw_to_cicids_"
            f"mmd_seed{seed}.pt"
        )
    )

    hda_checkpoint = torch.load(
        hda_checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    target_dim = int(
        hda_checkpoint[
            "target_dim"
        ]
    )

    target_encoder = TargetEncoder(
        input_dim=target_dim,
        latent_dim=168,
    )

    target_encoder.load_state_dict(
        hda_checkpoint[
            "target_encoder_state_dict"
        ]
    )

    target_model = TargetModel(
        target_encoder,
        source_model.classifier,
    )

    target_model.eval()

    # ======================================================
    # CICIDS TEST
    #
    # labels are used ONLY here.
    # ======================================================

    test_loader = make_loader(
        FEATURE_DIR
        / "cicids_test",
        input_dim=target_dim,
        batch_size=256,
        training=False,
    )

    labels, scores = collect_scores(
        target_model,
        test_loader,
    )
    
    print("\n--- SCORE DIAGNOSTICS ---")

    print(
        f"Score min:    {scores.min():.6f}"
    )

    print(
        f"Score max:    {scores.max():.6f}"
    )

    print(
        f"Score mean:   {scores.mean():.6f}"
    )

    print(
        f"Score median: {np.median(scores):.6f}"
    )

    for q in [
        0.01,
        0.05,
        0.25,
        0.50,
        0.75,
        0.95,
        0.99,
    ]:
        print(
            f"q{q:.2f}: "
            f"{np.quantile(scores, q):.6f}"
        )

    # ======================================================
    # Threshold
    #
    # IMPORTANT:
    # Do NOT use CICIDS validation labels.
    #
    # Use threshold selected on source validation.
    # ======================================================

    source_result_path = (
        PROJECT_DIR
        / "results"
        / "baseline"
        / f"unsw_seed{seed}.json"
    )

    with source_result_path.open(
        encoding="utf-8"
    ) as file:

        source_result = json.load(
            file
        )

    threshold = float(
        source_result["threshold"]
    )

    # For your current baseline:
    #
    # threshold ≈ 0.9843

    metrics = compute_metrics(
        labels,
        scores,
        threshold,
    )

    # ======================================================
    # Save
    # ======================================================

    result = {
        "experiment":
            "HDA + marginal RBF-MMD",

        "source_domain":
            "UNSW",

        "target_domain":
            "CICIDS",

        "target_labels_train":
            0,

        "seed":
            seed,

        "threshold_source":
            threshold,

        **metrics,
    }
    
    normal_scores = scores[
    labels == 0
    ]

    attack_scores = scores[
        labels == 1
    ]

    print("\n--- CLASS SCORE DIAGNOSTICS ---")

    print(
        f"Normal mean: "
        f"{normal_scores.mean():.6f}"
    )

    print(
        f"Attack mean: "
        f"{attack_scores.mean():.6f}"
    )

    print(
        f"Normal median: "
        f"{np.median(normal_scores):.6f}"
    )

    print(
        f"Attack median: "
        f"{np.median(attack_scores):.6f}"
    )
    
    for threshold_debug in [
    0.1,
    0.3,
    0.5,
    0.7,
    0.9,
        0.9843,
    ]:
        predictions = (
            scores >= threshold_debug
        )

        positive_rate = (
            predictions.mean()
        )

    print(
        f"threshold="
        f"{threshold_debug:.4f} | "
        f"predicted attack="
        f"{positive_rate:.4f}"
    )

    output_dir = (
        PROJECT_DIR
        / "results"
        / "hda"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / (
            "unsw_to_cicids_"
            f"mmd_seed{seed}.json"
        )
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            result,
            file,
            indent=2,
        )

    print(
        "\nHDA UNSW -> CICIDS"
    )

    print(
        f"Threshold (UNSW val): "
        f"{threshold:.4f}"
    )

    print(
        f"AP={metrics['pr_auc']:.4f} | "
        f"ROC-AUC="
        f"{metrics['roc_auc']:.4f} | "
        f"F1={metrics['f1']:.4f} | "
        f"Recall="
        f"{metrics['recall']:.4f} | "
        f"FPR={metrics['fpr']:.4f}"
    )

    print(
        f"Saved: {output_path}"
    )


if __name__ == "__main__":
    main()