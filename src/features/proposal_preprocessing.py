"""Fit on source train, then stream all proposal_v1 splits in fixed feature order."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from features.common_features import COMMON_FEATURES
from models.proposal_pipeline import proposal_processor

ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = ROOT / "data/features/common_raw"
OUTPUT_ROOT = ROOT / "data/features/proposal_v1"
BATCH_ROWS = 65536
INPUT_DIM = len(COMMON_FEATURES)


def direction_domains(direction):
    if direction == "unsw_to_cicids":
        return "unsw", "cicids"
    if direction == "cicids_to_unsw":
        return "cicids", "unsw"
    raise ValueError(f"Unknown direction: {direction}")


def parquet_files(directory):
    files = sorted(Path(directory).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet files in {directory}")
    return files


def read_batches(directory, include_labels=True):
    columns = ["features", "label"] if include_labels else ["features"]
    for path in parquet_files(directory):
        with pq.ParquetFile(path) as parquet:
            for batch in parquet.iter_batches(batch_size=BATCH_ROWS, columns=columns):
                vectors = batch.column("features")
                if not pa.types.is_fixed_size_list(vectors.type) or vectors.type.list_size != INPUT_DIM:
                    raise ValueError(f"Expected fixed {INPUT_DIM}-feature vectors in {path}")
                if vectors.null_count or vectors.values.null_count:
                    raise ValueError(f"Null feature vectors in {path}")
                values = vectors.values.to_numpy(zero_copy_only=False)
                features = np.asarray(values, dtype=np.float32).reshape(len(batch), INPUT_DIM)
                if include_labels:
                    labels = np.asarray(batch.column("label").to_numpy(zero_copy_only=False), dtype=np.int64)
                    yield features, labels
                else:
                    yield features


def fit_source_processor(source_train_path):
    files = parquet_files(source_train_path)
    total_rows = sum(pq.read_metadata(path).num_rows for path in files)
    source_train = np.empty((total_rows, INPUT_DIM), dtype=np.float32)
    offset = 0
    for batch in read_batches(source_train_path, include_labels=False):
        end = offset + len(batch)
        source_train[offset:end] = batch
        offset = end
    if offset != total_rows:
        raise RuntimeError("Source train row count changed while reading")
    processor = proposal_processor()
    processor.fit(source_train)
    return processor


def write_split(source_path, destination, processor):
    if destination.exists():
        raise FileExistsError(f"Prepared split already exists: {destination}")
    destination.mkdir(parents=True)
    output = destination / "data.parquet"
    vector_type = pa.list_(pa.float32(), INPUT_DIM)
    schema = pa.schema([("features", vector_type), ("label", pa.int64())])
    rows = 0
    with pq.ParquetWriter(output, schema, compression="snappy") as writer:
        for features, labels in read_batches(source_path):
            transformed = np.asarray(processor.transform(features), dtype=np.float32)
            if transformed.shape != features.shape or not np.isfinite(transformed).all():
                raise ValueError(f"Invalid transformed features in {source_path}")
            vectors = pa.FixedSizeListArray.from_arrays(
                pa.array(transformed.ravel(), type=pa.float32()), INPUT_DIM
            )
            writer.write_table(pa.table({"features": vectors, "label": pa.array(labels)}, schema=schema))
            rows += len(features)
    return rows


def prepare_direction(direction):
    source, target = direction_domains(direction)
    datasets = [(domain, split) for domain in (source, target) for split in ("train", "val", "test")]
    preprocessor_path = ROOT / "models/proposal_v1" / direction / "preprocessor.joblib"
    output_base = OUTPUT_ROOT / direction
    for path in [preprocessor_path, *(output_base / f"{domain}_{split}" for domain, split in datasets)]:
        if path.exists():
            raise FileExistsError(f"Proposal preprocessing output already exists: {path}")
    processor = fit_source_processor(RAW_ROOT / f"{source}_train")
    preprocessor_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(processor, preprocessor_path)
    print(f"Saved processor: {preprocessor_path}")
    for domain, split in datasets:
        rows = write_split(RAW_ROOT / f"{domain}_{split}", output_base / f"{domain}_{split}", processor)
        print(f"{direction} | {domain}_{split} | rows={rows} | features={INPUT_DIM}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direction", required=True, choices=["unsw_to_cicids", "cicids_to_unsw"])
    args = parser.parse_args()
    prepare_direction(args.direction)


if __name__ == "__main__":
    main()
