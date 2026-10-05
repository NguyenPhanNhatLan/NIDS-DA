import argparse
import hashlib
import json
from pathlib import Path

from features.common_features import (
    COMMON_FEATURES,
    DEFAULT_CONFIG,
    load_common_feature_config,
)

PROTOCOL = "proposal_v2"
DATA_REVISION = "spark_data_v1"
ROOT = Path(__file__).resolve().parents[2]


def mapping(domain):
    if domain not in ("unsw", "cicids"):
        raise ValueError(f"Unknown domain: {domain}")
    config = load_common_feature_config()
    if (
        len(config) != 5
        or tuple(item["canonical_name"] for item in config) != COMMON_FEATURES
    ):
        raise ValueError("Expected the canonical five-feature proposal_v2 schema")
    return config


def require_columns(frame, columns):
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def schema_hash():
    return hashlib.sha256(DEFAULT_CONFIG.read_bytes()).hexdigest()


def write_parquet(frame, output):
    frame.write.mode("errorifexists").option("compression", "snappy").parquet(
        str(output)
    )


def write_report(spark, report, output):
    payload = json.dumps(report, allow_nan=False, sort_keys=True)
    spark.createDataFrame([(payload,)], ["value"]).write.mode("errorifexists").text(
        str(output)
    )


def stage_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--domain", required=True, choices=["unsw", "cicids"])
    return parser
