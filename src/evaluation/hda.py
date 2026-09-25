import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

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


def diagnose_batch_norm(model, loader):
    """So sánh BN running stats và BN batch stats, không cập nhật model."""
    batch_norms = [
        layer for layer in model.encoder.modules()
        if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm)
    ]
    if not batch_norms:
        print("\n--- BN MODE DIAGNOSTIC ---")
        print("Encoder không có BatchNorm.")
        return

    # Sao lưu running_mean, running_var, num_batches_tracked.
    saved_buffers = [
        {name: value.clone() for name, value in layer.named_buffers()}
        for layer in batch_norms
    ]
    device = next(model.parameters()).device
    all_labels = []
    eval_scores = []
    batch_scores = []
    skipped_rows = 0

    try:
        with torch.no_grad():
            for features, labels in loader:
                # BN train mode không nhận batch chỉ có một dòng.
                if len(labels) < 2:
                    skipped_rows += len(labels)
                    continue

                features = features.to(device)
                # Mỗi batch eval dùng đúng running stats từ checkpoint.
                for layer, original in zip(batch_norms, saved_buffers):
                    for name, value in layer.named_buffers():
                        value.copy_(original[name])
                model.eval()
                _, eval_logits = model(features)

                # Dropout vẫn ở eval; chỉ BN dùng thống kê của batch hiện tại.
                for layer in batch_norms:
                    layer.train()
                _, batch_logits = model(features)

                all_labels.append(labels.numpy())
                eval_scores.append(
                    torch.softmax(eval_logits, dim=1)[:, 1].cpu().numpy()
                )
                batch_scores.append(
                    torch.softmax(batch_logits, dim=1)[:, 1].cpu().numpy()
                )
    finally:
        # Chạy BN ở train mode có thể thay đổi buffers dù đã no_grad.
        for layer, original in zip(batch_norms, saved_buffers):
            for name, value in layer.named_buffers():
                value.copy_(original[name])
        model.eval()

    if not all_labels:
        print("Không có batch >= 2 dòng cho BN diagnostic.")
        return

    labels = np.concatenate(all_labels)
    eval_scores = np.concatenate(eval_scores)
    batch_scores = np.concatenate(batch_scores)
    print("\n--- BN MODE DIAGNOSTIC (không dùng cho report) ---")
    print(f"Đã bỏ {skipped_rows} dòng ở batch cuối có kích thước 1.")
    print(f"Eval-mode score mean: {eval_scores.mean():.6f}")
    print(f"Batch-stat score mean: {batch_scores.mean():.6f}")

    if (labels == 1).any():
        print(f"Eval-mode attack mean: {eval_scores[labels == 1].mean():.6f}")
        print(f"Batch-stat attack mean: {batch_scores[labels == 1].mean():.6f}")
    if len(np.unique(labels)) == 2:
        print(f"Batch-stat AP: {average_precision_score(labels, batch_scores):.4f}")
        print(f"Batch-stat ROC-AUC: {roc_auc_score(labels, batch_scores):.4f}")
    else:
        print("AP/ROC-AUC cần cả hai lớp; không tính cho diagnostic này.")


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
    diagnose_batch_norm(target_model, test_loader)
    
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