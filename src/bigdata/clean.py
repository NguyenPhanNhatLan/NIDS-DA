from functools import reduce
from pyspark.sql import functions as F
from bigdata.common import mapping, require_columns, stage_parser, write_parquet
from spark_session import get_spark


def finite(value):
    return value.isNotNull() & ~F.isnan(value) & (F.abs(value) != float("inf"))


def clean(frame, domain):
    config = mapping(domain)
    require_columns(frame, [item[domain] for item in config] + ["label"])
    identity = [name for name in frame.columns if name != "_source_file"]
    frame = frame.dropDuplicates(identity)
    label = F.trim(F.col("label").cast("string"))
    if domain == "unsw":
        label_number = label.cast("double")
        binary = F.when(label_number.isin(0.0, 1.0), label_number.cast("long"))
    else:
        binary = F.when(F.upper(label) == "BENIGN", F.lit(0)).when(
            label.isNotNull() & (label != "") & ~label.rlike(r"^[+-]?\d+(\.\d+)?$"),
            F.lit(1),
        )
    frame = frame.withColumn("_binary_label", binary)
    if frame.filter(F.col("_binary_label").isNull()).limit(1).count():
        raise ValueError(
            f"Invalid {domain} raw labels; expected binary UNSW or named CICIDS categories"
        )
    expressions, invalid = [], []
    for item in config:
        value = F.trim(F.col(item[domain]).cast("string")).cast("double")
        valid = finite(value) & (value >= 0)
        if item["canonical_name"] != "flow_duration":
            valid = valid & (value == F.floor(value))
        expressions.append(
            F.when(valid, value)
            .otherwise(F.lit(None).cast("double"))
            .alias(item[domain])
        )
        invalid.append(~F.coalesce(valid, F.lit(False)))
    return frame.select(
        *expressions,
        F.col("_binary_label").cast("long").alias("label"),
        reduce(lambda a, b: a | b, invalid).alias("_invalid_features"),
    )


def validate_clean(frame, domain):
    columns = [item[domain] for item in mapping(domain)]
    require_columns(frame, columns + ["label"])
    invalid_label = F.col("label").isNull() | ~F.col("label").isin(0, 1)
    invalid_feature = reduce(
        lambda a, b: a | b,
        [
            F.col(name).isNotNull() & (~finite(F.col(name)) | (F.col(name) < 0))
            for name in columns
        ],
    )
    if frame.filter(invalid_label | invalid_feature).limit(1).count():
        raise ValueError("Clean schema has invalid feature values or labels")
    return frame


def main():
    args = stage_parser(__doc__).parse_args()
    spark = get_spark()
    try:
        write_parquet(
            validate_clean(
                clean(spark.read.parquet(args.input), args.domain), args.domain
            ),
            args.output,
        )
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
