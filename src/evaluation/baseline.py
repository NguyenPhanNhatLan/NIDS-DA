from pathlib import Path

import argparse
import json

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from models.baseline import BaselineMLP


PROJECT_DIR = Path(__file__).resolve().parents[2]


def evaluate(model, loader, domain_name):
    device = next(model.parameters()).device
    model.eval()

    all_labels = []
    all_predictions = []
    all_scores = []

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)

            _, logits = model(features)

            probabilities = torch.softmax(logits, dim=1)
            predictions = logits.argmax(dim=1)

            all_labels.append(labels.cpu().numpy())
            all_predictions.append(predictions.cpu().numpy())
            all_scores.append(probabilities[:, 1].cpu().numpy())

    if not all_labels:
        raise ValueError(f"Tập {domain_name} không có dữ liệu.")

    labels = np.concatenate(all_labels)
    predictions = np.concatenate(all_predictions)
    scores = np.concatenate(all_scores)

    print(f"\nKẾT QUẢ: {domain_name}")

    print(classification_report(
        labels,
        predictions,
        labels=[0, 1],
        target_names=["Normal", "Attack"],
        digits=4,
        zero_division=0,
    ))

    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(matrix)

    if len(np.unique(labels)) == 2:
        auc = roc_auc_score(labels, scores)
        print(f"ROC-AUC: {auc:.4f}")
    else:
        print("Không tính được ROC-AUC vì tập đánh giá chỉ có một lớp.")

def collect_scores(
    model,
    loader,
):
    device = next(
        model.parameters()
    ).device

    model.eval()

    all_labels = []
    all_scores = []

    with torch.no_grad():
        for features, labels in loader:

            features = features.to(
                device
            )

            _, logits = model(
                features
            )

            probabilities = torch.softmax(
                logits,
                dim=1,
            )[:, 1]

            all_labels.append(
                labels.numpy()
            )

            all_scores.append(
                probabilities
                .cpu()
                .numpy()
            )

    if not all_labels:
        raise ValueError("Evaluation loader is empty.")

    return (
        np.concatenate(
            all_labels
        ),
        np.concatenate(
            all_scores
        ),
    )


def select_f1_threshold(
    labels,
    scores,
):
    precision, recall, thresholds = (
        precision_recall_curve(
            labels,
            scores,
        )
    )

    if len(thresholds) == 0:
        return 0.5

    precision = precision[:-1]
    recall = recall[:-1]

    denominator = (
        precision + recall
    )

    f1 = np.divide(
        2 * precision * recall,
        denominator,
        out=np.zeros_like(
            denominator
        ),
        where=denominator > 0,
    )

    best_index = int(
        np.argmax(f1)
    )

    return float(
        thresholds[best_index]
    )


def compute_metrics(
    labels,
    scores,
    threshold,
):
    predictions = (
        scores >= threshold
    ).astype(int)

    tn, fp, fn, tp = (
        confusion_matrix(
            labels,
            predictions,
            labels=[0, 1],
        ).ravel()
    )

    ap = average_precision_score(
        labels,
        scores,
    )

    if len(np.unique(labels)) == 2:
        roc_auc = roc_auc_score(
            labels,
            scores,
        )
    else:
        roc_auc = float("nan")

    precision = precision_score(
        labels,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        labels,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        labels,
        predictions,
        zero_division=0,
    )

    fpr = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0.0
    )

    prevalence = float(
        np.mean(labels)
    )

    return {
        "pr_auc": float(ap),
        "roc_auc": float(roc_auc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fpr),
        "prevalence": prevalence,
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def evaluate_ap(
    model,
    loader,
    device,
):
    labels, scores = collect_scores(model, loader)

    if len(np.unique(labels)) != 2:
        raise ValueError("Validation must contain both classes.")

    return float(average_precision_score(labels, scores))


def evaluate_dataset(dataset, seed=42):
    # Import trong hàm để training có thể dùng evaluate_ap mà không bị import vòng.
    from training.baseline import make_loader

    checkpoint_path = (
        PROJECT_DIR / "models" / "baselines" / f"{dataset}_seed{seed}.pt"
    )
    device = torch.device("cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint["dataset"] != dataset or checkpoint["seed"] != seed:
        raise ValueError(f"Checkpoint không khớp dataset/seed: {checkpoint_path}")

    input_dim = int(checkpoint["input_dim"])
    model = BaselineMLP(input_dim)
    model.load_state_dict(checkpoint["model_state_dict"])

    feature_dir = PROJECT_DIR / "data" / "features"
    validation_loader = make_loader(
        feature_dir / f"{dataset}_val", input_dim, training=False
    )
    test_loader = make_loader(
        feature_dir / f"{dataset}_test", input_dim, training=False
    )

    # Chọn threshold từ validation, sau đó cố định khi tính metrics test.
    val_labels, val_scores = collect_scores(model, validation_loader)
    if len(np.unique(val_labels)) != 2:
        raise ValueError(f"{dataset} validation phải có đủ hai lớp.")
    threshold = select_f1_threshold(val_labels, val_scores)
    test_labels, test_scores = collect_scores(model, test_loader)
    if len(np.unique(test_labels)) != 2:
        raise ValueError(f"{dataset} test phải có đủ hai lớp.")
    metrics = compute_metrics(test_labels, test_scores, threshold)

    row = {
        "experiment": (
            "Source supervised" if dataset == "unsw"
            else "Target supervised reference"
        ),
        "feature_space": f"{dataset.upper()} full",
        "train_domain": dataset.upper(),
        "test_domain": dataset.upper(),
        "seed": seed,
        "best_epoch": checkpoint["best_epoch"],
        "best_val_ap": checkpoint["best_val_ap"],
        **metrics,
    }
    result_dir = PROJECT_DIR / "results" / "baseline"
    result_dir.mkdir(parents=True, exist_ok=True)
    output_path = result_dir / f"{dataset}_seed{seed}.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(row, file, indent=2)

    print(f"{row['experiment']} | seed={seed} | threshold={threshold:.4f}")
    print(f"AP={row['pr_auc']:.4f} | ROC-AUC={row['roc_auc']:.4f} | "
          f"F1={row['f1']:.4f} | Recall={row['recall']:.4f} | FPR={row['fpr']:.4f}")
    print(f"Saved: {output_path}")
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["unsw", "cicids"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    evaluate_dataset(args.dataset, args.seed)


if __name__ == "__main__":
    main()
