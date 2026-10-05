"""Select five flow features, standardize units, and retain scalar Spark columns."""

from pyspark.sql import functions as F
from bigdata.clean import validate_clean
from bigdata.common import mapping, stage_parser, write_parquet
from spark_session import get_spark


def harmonize(frame, domain):
    validate_clean(frame, domain)
    expressions = []
    for item in mapping(domain):
        value = F.col(item[domain]).cast("double")
        if item["transform"] == "cicids_us_to_seconds":
            if domain == "cicids":
                value = value / 1_000_000.0
        elif item["transform"] != "none":
            raise ValueError(f'Unsupported v2 transform: {item["transform"]}')
        expressions.append(value.alias(item["canonical_name"]))
    return frame.select(*expressions, F.col("label").cast("long"))


def main():
    args = stage_parser(__doc__).parse_args()
    spark = get_spark()
    try:
        write_parquet(
            harmonize(spark.read.parquet(args.input), args.domain), args.output
        )
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
