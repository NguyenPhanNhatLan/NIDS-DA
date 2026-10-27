import hashlib
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from features.common_features import COMMON_FEATURES
from features.parquet_vectors import vector_matrix

INPUT_DIM = len(COMMON_FEATURES)


def split_sha256(path):
    files = sorted(Path(path).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet files in {path}")
    digest = hashlib.sha256()
    for parquet_path in files:
        digest.update(parquet_path.name.encode())
        with parquet_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def iter_parquet_batches(
    path, batch_size, shuffle, seed, include_labels, drop_last=False
):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    files = sorted(Path(path).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet files in {path}")
    rng = np.random.default_rng(seed)
    if shuffle:
        rng.shuffle(files)
    columns = ["features", "label"] if include_labels else ["features"]
    pending_features = np.empty((0, INPUT_DIM), dtype=np.float32)
    pending_labels = np.empty(0, dtype=np.int64)
    for parquet_path in files:
        with pq.ParquetFile(parquet_path) as parquet:
            for batch in parquet.iter_batches(
                batch_size=max(8192, batch_size), columns=columns
            ):
                vectors = batch.column("features")
                feature_values = vector_matrix(vectors, parquet_path)
                if include_labels:
                    label_values = np.asarray(
                        batch.column("label").to_numpy(zero_copy_only=False),
                        dtype=np.int64,
                    )
                    if not np.isin(label_values, [0, 1]).all():
                        raise ValueError(f"Invalid labels in {parquet_path}")
                if shuffle:
                    order = rng.permutation(len(feature_values))
                    feature_values = feature_values[order]
                    if include_labels:
                        label_values = label_values[order]
                if len(pending_features):
                    feature_values = np.concatenate((pending_features, feature_values))
                    if include_labels:
                        label_values = np.concatenate((pending_labels, label_values))
                complete_rows = len(feature_values) // batch_size * batch_size
                for start in range(0, complete_rows, batch_size):
                    features = torch.from_numpy(
                        np.array(feature_values[start : start + batch_size], copy=True)
                    )
                    if include_labels:
                        labels = torch.from_numpy(
                            np.array(label_values[start : start + batch_size], copy=True)
                        )
                        yield features, labels
                    else:
                        yield features
                pending_features = feature_values[complete_rows:].copy()
                if include_labels:
                    pending_labels = label_values[complete_rows:].copy()
    if len(pending_features) and not drop_last:
        features = torch.from_numpy(np.array(pending_features, copy=True))
        if include_labels:
            yield features, torch.from_numpy(np.array(pending_labels, copy=True))
        else:
            yield features


class ParquetBatchStream:
    def __init__(
        self, path, batch_size, shuffle, seed, include_labels, drop_last=False
    ):
        self.path = path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.include_labels = include_labels
        self.drop_last = drop_last
        self.epoch = 0

    def __iter__(self):
        seed = self.seed + self.epoch if self.shuffle else self.seed
        if self.shuffle:
            self.epoch += 1
        return iter_parquet_batches(
            self.path,
            self.batch_size,
            self.shuffle,
            seed,
            self.include_labels,
            self.drop_last,
        )
