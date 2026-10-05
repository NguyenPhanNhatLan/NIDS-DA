"""Small deterministic Parquet batch stream for proposal_v2 training."""

import hashlib
from pathlib import Path

import numpy as np
import pyarrow as pa
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
    for file in files:
        digest.update(file.name.encode())
        with file.open("rb") as stream:
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
    pending_x = np.empty((0, INPUT_DIM), dtype=np.float32)
    pending_y = np.empty(0, dtype=np.int64)
    for file in files:
        with pq.ParquetFile(file) as parquet:
            for batch in parquet.iter_batches(
                batch_size=max(8192, batch_size), columns=columns
            ):
                vectors = batch.column("features")
                x = vector_matrix(vectors, file)
                if include_labels:
                    y = np.asarray(
                        batch.column("label").to_numpy(zero_copy_only=False),
                        dtype=np.int64,
                    )
                    if not np.isin(y, [0, 1]).all():
                        raise ValueError(f"Invalid labels in {file}")
                if shuffle:
                    order = rng.permutation(len(x))
                    x = x[order]
                    if include_labels:
                        y = y[order]
                if len(pending_x):
                    x = np.concatenate((pending_x, x))
                    if include_labels:
                        y = np.concatenate((pending_y, y))
                full = len(x) // batch_size * batch_size
                for start in range(0, full, batch_size):
                    features = torch.from_numpy(
                        np.array(x[start : start + batch_size], copy=True)
                    )
                    if include_labels:
                        labels = torch.from_numpy(
                            np.array(y[start : start + batch_size], copy=True)
                        )
                        yield features, labels
                    else:
                        yield features
                pending_x = x[full:].copy()
                if include_labels:
                    pending_y = y[full:].copy()
    if len(pending_x) and not drop_last:
        features = torch.from_numpy(np.array(pending_x, copy=True))
        if include_labels:
            yield features, torch.from_numpy(np.array(pending_y, copy=True))
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
