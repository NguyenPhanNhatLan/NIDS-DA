"""Read headered raw flow CSVs as strings without driver-side data loading."""

import argparse
import re
from pyspark.sql import functions as F
from bigdata.common import mapping, require_columns, write_parquet
from spark_session import get_spark


def normalize_headers(frame, domain):
    required = {item[domain] for item in mapping(domain)} | {"label"}
    used, names = set(), []
    for column in frame.columns:
        name = re.sub(r"[^a-z0-9_]+", "_", column.strip().lower()).strip("_")
        if not name:
            raise ValueError("Empty CSV column name")
        if name in used:
            if name in required:
                raise ValueError(f"Ambiguous required CSV column: {name}")
            suffix = 2
            while f"{name}_dup{suffix}" in used:
                suffix += 1
            name = f"{name}_dup{suffix}"
        used.add(name)
        names.append(name)
    normalized = frame.toDF(*names)
    require_columns(normalized, required)
    return normalized


def ingest(spark, inputs, domain):
    frame = (
        spark.read.option("header", True)
        .option("inferSchema", False)
        .option("enforceSchema", False)
        .option("mode", "FAILFAST")
        .csv([str(path) for path in inputs])
    )
    frame = normalize_headers(frame, domain)
    if "_source_file" in frame.columns:
        raise ValueError("Reserved input column: _source_file")
    return frame.withColumn("_source_file", F.input_file_name())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--domain", choices=["unsw", "cicids"], required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    spark = get_spark()
    try:
        write_parquet(ingest(spark, args.input, args.domain), args.output)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
