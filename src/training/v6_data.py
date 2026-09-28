from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, IterableDataset


class ParquetBatches(IterableDataset):
    def __init__(self, path, input_dim, batch_size, training=False, labeled=True):
        self.files = sorted(Path(path).glob("*.parquet"))
        if not self.files:
            raise FileNotFoundError(f"No Parquet files in {path}")
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.training = training
        self.labeled = labeled

    def __iter__(self):
        files = list(self.files)
        rng = np.random.default_rng(torch.randint(0, 2**32, ()).item())
        if self.training:
            rng.shuffle(files)
        pending_x = None
        pending_y = None
        columns = ["features", "label"] if self.labeled else ["features"]
        for path in files:
            with pq.ParquetFile(path) as parquet:
                for batch in parquet.iter_batches(batch_size=8192, columns=columns):
                    values = batch.column("features")
                    lengths = pc.list_value_length(values).to_numpy(zero_copy_only=False)
                    if values.null_count or not np.all(lengths == self.input_dim):
                        raise ValueError(f"Wrong feature dimensions in {path}")
                    features = values.flatten().to_numpy(zero_copy_only=False)
                    features = np.array(features, dtype=np.float32).reshape(len(values), self.input_dim)
                    if not np.isfinite(features).all():
                        raise ValueError(f"Features contain NaN or Inf in {path}")
                    labels = None
                    if self.labeled:
                        labels = batch.column("label").to_numpy(zero_copy_only=False)
                        if not np.isin(labels, [0, 1]).all():
                            raise ValueError(f"Labels must be 0 or 1 in {path}")
                        labels = torch.from_numpy(np.array(labels, dtype=np.int64))
                    indices = np.arange(len(features))
                    if self.training:
                        rng.shuffle(indices)
                    features = torch.from_numpy(features[indices])
                    if self.labeled:
                        labels = labels[indices]
                    if pending_x is not None:
                        features = torch.cat((pending_x, features))
                        if self.labeled:
                            labels = torch.cat((pending_y, labels))
                    full_rows = len(features) // self.batch_size * self.batch_size
                    for start in range(0, full_rows, self.batch_size):
                        end = start + self.batch_size
                        if self.labeled:
                            yield features[start:end], labels[start:end]
                        else:
                            yield features[start:end]
                    pending_x = features[full_rows:]
                    pending_y = labels[full_rows:] if self.labeled else None
        if not self.training and pending_x is not None and len(pending_x):
            yield (pending_x, pending_y) if self.labeled else pending_x


def make_loader(path, input_dim, batch_size=256, training=False):
    dataset = ParquetBatches(path, input_dim, batch_size, training, labeled=True)
    return DataLoader(dataset, batch_size=None, num_workers=0)


def make_unlabeled_loader(path, input_dim, batch_size=256):
    dataset = ParquetBatches(path, input_dim, batch_size, training=True, labeled=False)
    return DataLoader(dataset, batch_size=None, num_workers=0)


def make_teacher_loader(path, input_dim, batch_size):
    dataset = ParquetBatches(path, input_dim, batch_size, training=False, labeled=False)
    return DataLoader(dataset, batch_size=None, num_workers=0)
