import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, IterableDataset

from models.baseline import BaselineMLP

PROJECT_DIR = Path(__file__).resolve().parents[2]
FEATURE_DIR = PROJECT_DIR / "data" / "features"


def train_baseline(model, counts, train_loader, val_loader, epochs=20, lr=0.001):
    device = next(model.parameters()).device

    counts = torch.as_tensor(counts, dtype=torch.float32)

    if (counts == 0).any():
        raise ValueError("Training data must contain both classes.")

    # Lớp ít mẫu hơn sẽ có trọng số lớn hơn.
    class_weights = counts.sum() / (2 * counts)
    class_weights = class_weights.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    print("-- Start Training--")
    for epoch in range(epochs):
        model.train()
        loss_sum = 0.0
        weight_sum = 0.0

        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            _, logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            batch_weight = class_weights[y].sum().item()
            loss_sum += loss.item() * batch_weight
            weight_sum += batch_weight

        if weight_sum == 0:
            raise ValueError("No training batches; check dataset size and batch_size.")

        f1 = evaluate_f1(model, val_loader, device)
        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Loss: {loss_sum / weight_sum:.4f} | "
            f"Val attack F1: {f1:.4f}"
        )

    return model


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
            raise FileNotFoundError(f"No Parquet files in {path}. Run feature engineering first.")
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
    torch.manual_seed(42)

    # 1. Đọc thông tin đã lưu ở bước feature engineering.
    with (FEATURE_DIR / "metadata.json").open(encoding="utf-8") as f:
        metadata = json.load(f)
    input_dim = int(metadata["input_dim"])
    normal_count = metadata["class_counts"]["0"]
    attack_count = metadata["class_counts"]["1"]
    counts = [normal_count, attack_count]

    # 2. Tạo bộ đọc dữ liệu theo batch.
    train_loader = make_loader(FEATURE_DIR / "unsw_train", input_dim, training=True)
    val_loader = make_loader(FEATURE_DIR / "unsw_val", input_dim)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # 3. Khởi tạo và huấn luyện mô hình.
    model = BaselineMLP(input_dim=input_dim).to(device)
    model = train_baseline(model, counts, train_loader, val_loader)

    # 4. Lưu trọng số và số chiều đầu vào để dùng khi dự đoán.
    model_dir = PROJECT_DIR / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"input_dim": input_dim, "model_state_dict": model.cpu().state_dict()},
        model_dir / "unsw_mlp.pt",
    )


if __name__ == "__main__":
    main()
