from copy import deepcopy
import random
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, IterableDataset

from evaluation.baseline import evaluate_ap
from models.baseline import BaselineMLP

PROJECT_DIR = Path(__file__).resolve().parents[2]
FEATURE_DIR = PROJECT_DIR / "data" / "features"


def train_baseline(
    model,
    counts,
    train_loader,
    val_loader,
    epochs=50,
    lr=0.001,
    patience=7,
    min_delta=1e-4,
):
    device = next(model.parameters()).device

    counts = torch.as_tensor(
        counts,
        dtype=torch.float32,
    )

    if (counts == 0).any():
        raise ValueError("Training data must " "contain both classes.")

    class_weights = (counts.sum() / (2 * counts)).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4,
    )

    best_ap = -np.inf
    best_epoch = -1
    best_state = None

    epochs_without_improvement = 0

    print("-- Start Training --")

    for epoch in range(
        1,
        epochs + 1,
    ):
        model.train()

        loss_sum = 0.0
        batches = 0

        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            _, logits = model(X)

            loss = criterion(
                logits,
                y,
            )

            loss.backward()

            optimizer.step()

            loss_sum += loss.item()
            batches += 1

        if batches == 0:
            raise ValueError("No training batches.")

        val_ap = evaluate_ap(
            model,
            val_loader,
            device,
        )

        train_loss = loss_sum / batches

        print(
            f"Epoch {epoch:03d} | " f"Loss={train_loss:.5f} | " f"Val AP={val_ap:.5f}"
        )

        improvement = val_ap - best_ap

        if val_ap > best_ap:
            best_ap = val_ap
            best_epoch = epoch

            best_state = deepcopy(model.state_dict())

        if improvement > min_delta:
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print("Early stopping.")
            break

    if best_state is None:
        raise RuntimeError("No checkpoint selected.")

    model.load_state_dict(best_state)

    print(f"Best epoch={best_epoch} | " f"Best validation AP={best_ap:.5f}")

    return (
        model,
        best_epoch,
        best_ap,
    )


def set_seed(seed):
    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_f1(model, val_loader, device):
    """Tính F1 của lớp tấn công (label = 1) trên toàn bộ tập validation."""
    model.eval()
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    with torch.no_grad():
        for features, labels in val_loader:
            features = features.to(device)
            labels = labels.to(device)
            _, logits = model(features)
            predictions = logits.argmax(dim=1)

            true_positives += ((predictions == 1) & (labels == 1)).sum().item()
            false_positives += ((predictions == 1) & (labels == 0)).sum().item()
            false_negatives += ((predictions == 0) & (labels == 1)).sum().item()

    denominator = 2 * true_positives + false_positives + false_negatives
    if denominator == 0:
        return 0.0
    return 2 * true_positives / denominator


class ParquetRows(IterableDataset):
    """Đọc tối đa 8192 dòng mỗi lần, sau đó đưa từng dòng cho DataLoader."""

    def __init__(self, path, input_dim, training=False):
        self.files = sorted(Path(path).glob("*.parquet"))
        if not self.files:
            raise FileNotFoundError(
                f"No Parquet files in {path}. Run feature engineering first."
            )
        self.input_dim = input_dim
        self.training = training

    def __iter__(self):
        files = list(self.files)
        seed = torch.randint(0, 2**32, ()).item()
        rng = np.random.default_rng(seed)
        if self.training:
            rng.shuffle(files)
        for path in files:
            with pq.ParquetFile(path) as parquet:
                chunks = parquet.iter_batches(
                    batch_size=8192, columns=["features", "label"]
                )
                for batch in chunks:
                    feature_rows = batch.column("features").to_pylist()
                    features = np.asarray(feature_rows, dtype=np.float32)
                    labels = batch.column("label").to_pylist()
                    if features.ndim != 2 or features.shape[1] != self.input_dim:
                        raise ValueError(f"Wrong feature dimensions in {path}")
                    if not np.isfinite(features).all():
                        raise ValueError(f"Features contain NaN or Inf in {path}")
                    if any(label not in (0, 1) for label in labels):
                        raise ValueError(f"Labels must be 0 or 1 in {path}")
                    features = torch.from_numpy(features)
                    labels = torch.tensor(labels, dtype=torch.long)

                    # Chỉ xáo trộn trong từng phần đang đọc, không nạp cả tập vào RAM.
                    row_indices = np.arange(len(labels))
                    if self.training:
                        rng.shuffle(row_indices)

                    for index in row_indices:
                        yield features[index], labels[index]


def make_loader(path, input_dim, batch_size=256, training=False):
    dataset = ParquetRows(path, input_dim, training)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        # Bỏ batch cuối chưa đủ kích thước khi train để tránh BatchNorm nhận 1 mẫu.
        drop_last=training,
    )


def main():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        choices=["unsw", "cicids"],
        required=True,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    dataset = args.dataset
    seed = args.seed

    set_seed(seed)

    metadata_path = FEATURE_DIR / f"{dataset}_metadata.json"

    if not metadata_path.exists() or metadata_path.stat().st_size == 0:
        raise FileNotFoundError(
            f"Metadata thiếu hoặc rỗng: {metadata_path}. "
            f"Chạy lại feature_engineering cho {dataset} trước khi train."
        )

    with metadata_path.open(encoding="utf-8") as f:
        metadata = json.load(f)

    input_dim = int(metadata["input_dim"])

    normal_count = int(metadata["class_counts"]["0"])

    attack_count = int(metadata["class_counts"]["1"])

    counts = [
        normal_count,
        attack_count,
    ]

    train_loader = make_loader(
        FEATURE_DIR / f"{dataset}_train",
        input_dim,
        training=True,
    )

    val_loader = make_loader(
        FEATURE_DIR / f"{dataset}_val",
        input_dim,
        training=False,
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")

    elif torch.backends.mps.is_available():
        device = torch.device("mps")

    else:
        device = torch.device("cpu")

    print(f"Dataset: {dataset}")

    print(f"Seed: {seed}")

    print(f"Device: {device}")

    print(f"Input dim: {input_dim}")

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
    )

    model_dir = PROJECT_DIR / "models" / "baselines"

    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = model_dir / f"{dataset}_seed{seed}.pt"

    torch.save(
        {
            "dataset": dataset,
            "seed": seed,
            "input_dim": input_dim,
            "best_epoch": best_epoch,
            "best_val_ap": best_val_ap,
            "model_state_dict": model.cpu().state_dict(),
        },
        output_path,
    )

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
