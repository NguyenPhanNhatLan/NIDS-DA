"""Fit and write upstream directional preprocessing with Spark."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
from project_utils import domains as direction_domains


ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = ROOT / "data/features/common_raw2"
OUTPUT_ROOT = ROOT / "data/features/proposal_v2"


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
