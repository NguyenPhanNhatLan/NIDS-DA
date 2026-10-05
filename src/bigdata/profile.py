"""Distributed quality summaries; collect only scalar per-feature statistics."""

import argparse
from pyspark.sql import functions as F
from bigdata.common import PROTOCOL, DATA_REVISION, mapping, schema_hash, write_report
from bigdata.clean import validate_clean
from spark_session import get_spark


def profile(frame, domain, relative_error=0.001):
    if not 0 < relative_error <= 0.1:
        raise ValueError(
            "relative_error must be in (0, 0.1] for bounded quantile summaries"
        )
    validate_clean(frame, domain)
    names = [item[domain] for item in mapping(domain)]
    expressions = [
        F.count("*").alias("rows"),
        F.sum(F.when(F.col("label") == 1, 1).otherwise(0)).alias("attack_rows"),
    ]
    for i, name in enumerate(names):
        expressions.extend(
            [
                F.sum(F.when(F.col(name).isNull(), 1).otherwise(0)).alias(
                    f"missing_{i}"
                ),
                F.min(name).alias(f"min_{i}"),
                F.max(name).alias(f"max_{i}"),
            ]
        )
    values = frame.agg(*expressions).first().asDict()
    quantiles = frame.approxQuantile(names, [0.25, 0.5, 0.75], relative_error)
    return {
        "protocol": PROTOCOL,
        "data_revision": DATA_REVISION,
        "domain": domain,
        "common_feature_config_sha256": schema_hash(),
        "rows": values["rows"],
        "attack_rows": values["attack_rows"] or 0,
        "relative_error": relative_error,
        "features": {
            name: {
                "missing": values[f"missing_{i}"] or 0,
                "min": values[f"min_{i}"],
                "max": values[f"max_{i}"],
                "q25_median_q75": quantiles[i],
            }
            for i, name in enumerate(names)
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--domain", required=True, choices=["unsw", "cicids"])
    args = parser.parse_args()
    spark = get_spark()
    try:
        write_report(
            spark, profile(spark.read.parquet(args.input), args.domain), args.output
        )
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
