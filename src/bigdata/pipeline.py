import argparse
import json
from pathlib import Path
from datetime import datetime, timezone
from pyspark.sql import functions as F
from bigdata.aggregate import aggregate
from bigdata.clean import clean, validate_clean
from bigdata.common import (
    ROOT,
    PROTOCOL,
    DATA_REVISION,
    schema_hash,
    write_parquet,
    write_report,
)
from bigdata.harmonize import harmonize
from bigdata.ingest import ingest
from bigdata.profile import profile
from bigdata.split import split
from features.proposal_preprocessing import prepare_direction
from spark_session import get_spark


def run(spark, inputs, work_root, feature_root, model_root, relative_error=0.001):
    roots = [Path(work_root), Path(feature_root), Path(model_root)]
    work_root, feature_root, model_root = roots
    for path in roots:
        if path.exists():
            raise FileExistsError(f"Use a fresh output root; already exists: {path}")
    if any(
        a == b or a in b.parents or b in a.parents
        for i, a in enumerate(roots)
        for b in roots[i + 1 :]
    ):
        raise ValueError(
            "Work, feature and model roots must be distinct, nonnested paths"
        )
    if set(inputs) != {"unsw", "cicids"}:
        raise ValueError("Both raw domains are required")
    frames = {}
    reports = {}
    for domain in ("unsw", "cicids"):
        raw_path = work_root / "ingested" / domain
        clean_path = work_root / "clean" / domain
        write_parquet(ingest(spark, inputs[domain], domain), raw_path)
        raw = spark.read.parquet(str(raw_path))
        write_parquet(validate_clean(clean(raw, domain), domain), clean_path)
        cleaned = spark.read.parquet(str(clean_path))
        report = profile(cleaned, domain, relative_error)
        report["input_rows"] = raw.count()
        # Count imported columns before feature selection; never count labels as features.
        imported_columns = [c for c in raw.columns if c != "_source_file"]
        label_columns = [c for c in imported_columns if c in {"label", "attack_cat"}]
        report["original_columns"] = imported_columns
        report["excluded_label_columns"] = label_columns
        report["original_feature_count"] = len(imported_columns) - len(label_columns)
        report["original_feature_definition"] = (
            "Imported non-label columns after header normalization and duplicate-header "
            "alias consolidation, before five-feature selection; identifiers included"
        )
        report["benign_rows"] = report["rows"] - report["attack_rows"]
        report["attack_ratio"] = report["attack_rows"] / report["rows"] if report["rows"] else None
        raw_label = F.trim(F.col("label").cast("string"))
        is_benign = raw_label.cast("double") == 0 if domain == "unsw" else F.upper(raw_label) == "BENIGN"
        input_benign = raw.agg(F.sum(F.when(is_benign, 1).otherwise(0))).first()[0] or 0
        report["input_benign_rows"] = input_benign
        report["input_attack_rows"] = report["input_rows"] - input_benign
        report["exact_duplicates_removed"] = report["input_rows"] - report["rows"]
        write_report(spark, report, work_root / "profiles" / domain)
        reports[domain] = report
        for name, part in split(cleaned, domain).items():
            split_path = work_root / "splits" / f"{domain}_{name}"
            common_path = work_root / "common" / f"{domain}_{name}"
            write_parquet(part, split_path)
            write_parquet(
                harmonize(spark.read.parquet(str(split_path)), domain), common_path
            )
            frames[(domain, name)] = spark.read.parquet(str(common_path))
    counts = aggregate(frames)
    write_report(spark, counts, work_root / "class_counts")
    for direction in ("unsw_to_cicids", "cicids_to_unsw"):
        prepare_direction(
            direction,
            spark=spark,
            raw_root=work_root / "common",
            output_root=feature_root,
            model_root=model_root,
            relative_error=relative_error,
        )
    manifest = {
        "protocol": PROTOCOL,
        "data_revision": DATA_REVISION,
        "spark_version": spark.version,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_statistics_stage": "clean_after_exact_deduplication_before_split",
        "relative_error": relative_error,
        "common_feature_config_sha256": schema_hash(),
        "raw_inputs": {
            domain: [str(p) for p in paths] for domain, paths in inputs.items()
        },
        "work_root": str(work_root),
        "feature_root": str(feature_root),
        "model_root": str(model_root),
        "split_seed": 42,
        "split_percentages": [70, 15, 15],
        "preprocessor_fit_role": "source_train_only",
        "deduplication": "exact full raw rows excluding source filename",
        "counts": counts,
        "profiles": reports,
    }

    with (work_root / "manifest.json").open("x") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unsw-csv", nargs="+", required=True)
    parser.add_argument("--cicids-csv", nargs="+", required=True)
    parser.add_argument(
        "--work-root", type=Path, default=ROOT / "data/bigdata/proposal_v2"
    )
    parser.add_argument(
        "--feature-root", type=Path, default=ROOT / "data/features/proposal_v2"
    )
    parser.add_argument("--model-root", type=Path, default=ROOT / "models/proposal_v2")
    parser.add_argument("--relative-error", type=float, default=0.001)
    args = parser.parse_args()
    spark = get_spark()
    try:
        run(
            spark,
            {"unsw": args.unsw_csv, "cicids": args.cicids_csv},
            args.work_root,
            args.feature_root,
            args.model_root,
            args.relative_error,
        )
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
