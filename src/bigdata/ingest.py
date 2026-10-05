"""Read headered raw flow CSVs as strings without driver-side data loading."""

import argparse
import re
from pyspark.sql import functions as F
from bigdata.common import mapping, require_columns, write_parquet
from spark_session import get_spark


def collapse_forward_header_length(frame):
    canonical, alias = "fwd_header_length", "fwd_header_length_1"
    if alias not in frame.columns:
        return frame
    if canonical not in frame.columns:
        return frame.withColumnRenamed(alias, canonical)

    def value(name):
        text = F.trim(F.col(name).cast("string"))
        return F.when(text != "", text)

    left, right = value(canonical), value(alias)
    same = (left == right) | (left.cast("double") == right.cast("double"))
    conflict = left.isNotNull() & right.isNotNull() & ~F.coalesce(same, F.lit(False))
    if frame.filter(conflict).limit(1).count():
        raise ValueError("Conflicting CICIDS fwd_header_length and fwd_header_length_1 values")
    return frame.withColumn(canonical, F.coalesce(left, right)).drop(alias)


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
    if domain == "cicids":
        normalized = collapse_forward_header_length(normalized)
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
