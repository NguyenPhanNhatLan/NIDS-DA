"""Distributed data counts per domain/split/class, separate from ML aggregate."""

import argparse
from functools import reduce
from pyspark.sql import functions as F
from bigdata.common import PROTOCOL, DATA_REVISION, write_report
from spark_session import get_spark


def aggregate(frames):
    parts = [
        frame.groupBy("label")
        .count()
        .withColumn("domain", F.lit(domain))
        .withColumn("split", F.lit(split))
        for (domain, split), frame in frames.items()
    ]
    if not parts:
        raise ValueError("No split frames")
    rows = (
        reduce(lambda a, b: a.unionByName(b), parts)
        .orderBy("domain", "split", "label")
        .collect()
    )
    return {
        "protocol": PROTOCOL,
        "data_revision": DATA_REVISION,
        "class_counts": [row.asDict() for row in rows],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    spark = get_spark()
    try:
        frames = {
            (domain, split): spark.read.parquet(
                f'{args.input_root.rstrip("/")}/{domain}_{split}'
            )
            for domain in ("unsw", "cicids")
            for split in ("train", "val", "test")
        }
        write_report(spark, aggregate(frames), args.output)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
