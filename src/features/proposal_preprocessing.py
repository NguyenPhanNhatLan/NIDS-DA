"""Fit on source train, then stream all proposal_v2 splits in fixed feature order."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from features.common_features import COMMON_FEATURES
from features.parquet_vectors import vector_matrix

ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = ROOT / "data/features/common_raw2"
OUTPUT_ROOT = ROOT / "data/features/proposal_v2"
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
                features = vector_matrix(vectors, path, allow_nonfinite=True)
                if include_labels:
                    labels = np.asarray(
                        batch.column("label").to_numpy(zero_copy_only=False),
                        dtype=np.int64,
                    )
                    yield features, labels
                else:
                    yield features


def fit_source_processor(source_train_path, spark=None, relative_error=0.001):
    """Fit distributed quantile summaries; no source matrix on the driver."""
    from bigdata.preprocess import fit_source_processor as spark_fit
    from spark_session import get_spark

    owned = spark is None
    spark = spark or get_spark()
    try:
        return spark_fit(spark.read.parquet(str(source_train_path)), relative_error)
    finally:
        if owned:
            spark.stop()


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
            if (
                transformed.shape != features.shape
                or not np.isfinite(transformed).all()
            ):
                raise ValueError(f"Invalid transformed features in {source_path}")
            vectors = pa.FixedSizeListArray.from_arrays(
                pa.array(transformed.ravel(), type=pa.float32()), INPUT_DIM
            )
            writer.write_table(
                pa.table(
                    {"features": vectors, "label": pa.array(labels)}, schema=schema
                )
            )
            rows += len(features)
    return rows


def prepare_direction(
    direction,
    spark=None,
    raw_root=None,
    output_root=None,
    model_root=None,
    relative_error=0.001,
):
    """Spark fits on source train and transforms every split in executor partitions."""
    from bigdata.preprocess import write_prepared
    from spark_session import get_spark

    source, target = direction_domains(direction)
    raw_root = Path(raw_root) if raw_root is not None else RAW_ROOT
    output_root = Path(output_root) if output_root is not None else OUTPUT_ROOT
    model_root = (
        Path(model_root) if model_root is not None else ROOT / "models/proposal_v2"
    )
    datasets = [
        (domain, split)
        for domain in (source, target)
        for split in ("train", "val", "test")
    ]
    preprocessor_path = model_root / direction / "preprocessor.joblib"
    output_base = output_root / direction
    for path in [
        preprocessor_path,
        *(output_base / f"{domain}_{split}" for domain, split in datasets),
    ]:
        if path.exists():
            raise FileExistsError(
                f"Proposal preprocessing output already exists: {path}"
            )
    owned = spark is None
    spark = spark or get_spark()
    try:
        processor = fit_source_processor(
            raw_root / f"{source}_train", spark, relative_error
        )
        preprocessor_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(processor, preprocessor_path)
        for domain, split in datasets:
            frame = spark.read.parquet(str(raw_root / f"{domain}_{split}"))
            write_prepared(frame, output_base / f"{domain}_{split}", processor)
        return processor
    finally:
        if owned:
            spark.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--direction", required=True, choices=["unsw_to_cicids", "cicids_to_unsw"]
    )
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--model-root", type=Path, default=ROOT / "models/proposal_v2")
    parser.add_argument("--relative-error", type=float, default=0.001)
    args = parser.parse_args()
    prepare_direction(
        args.direction,
        raw_root=args.raw_root,
        output_root=args.output_root,
        model_root=args.model_root,
        relative_error=args.relative_error,
    )


if __name__ == "__main__":
    main()
