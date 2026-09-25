from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from models.baseline import BaselineMLP
from training.baseline import make_loader


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


def main():
    # Dùng CPU để đoạn code chạy được trên cả Mac và máy không có GPU.
    device = torch.device("cpu")

    checkpoint = torch.load(
        PROJECT_DIR / "models" / "unsw_mlp.pt",
        map_location=device,
        weights_only=True,
    )

    input_dim = checkpoint["input_dim"]

    model = BaselineMLP(input_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    source_loader = make_loader(
        PROJECT_DIR / "data" / "features" / "unsw_test",
        input_dim=input_dim,
        training=False,
    )

    evaluate(model, source_loader, "Source - UNSW")

    # Chỉ bật phần này SAU KHI chuẩn bị bộ features chung
    # và huấn luyện lại mô hình trên bộ features đó.
    #
    # target_loader = make_loader(
    #     PROJECT_DIR / "data" / "features" / "cicids_test",
    #     input_dim=input_dim,
    #     training=False,
    # )
    # evaluate(model, target_loader, "Target - CICIDS")


if __name__ == "__main__":
    main()