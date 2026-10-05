"""Fit source-only median/log/IQR in Spark; write distributed five-float arrays."""

import math
from pyspark.sql import functions as F
from bigdata.clean import finite
from bigdata.common import require_columns, schema_hash, write_parquet
from features.common_features import COMMON_FEATURES
from models.proposal_pipeline import DistributedSourceProcessor


def scalar_features(frame):
    if set(COMMON_FEATURES) <= set(frame.columns):
        selected = frame.select(*COMMON_FEATURES)
    elif "features" in frame.columns:
        if (
            frame.filter(F.col("features").isNull() | (F.size("features") != 5))
            .limit(1)
            .count()
        ):
            raise ValueError("Expected five-element feature arrays")
        selected = frame.select(
            *[
                F.col("features")[i].alias(name)
                for i, name in enumerate(COMMON_FEATURES)
            ]
        )
    else:
        raise ValueError("Missing canonical features")
    return selected.select(
        *[
            F.when(
                finite(F.col(name).cast("double")), F.col(name).cast("double")
            ).alias(name)
            for name in COMMON_FEATURES
        ]
    )


def log_imputed(frame, medians):
    return frame.select(
        *[
            (
                F.signum(F.coalesce(F.col(name), F.lit(median)))
                * F.log1p(F.abs(F.coalesce(F.col(name), F.lit(median))))
            ).alias(name)
            for name, median in zip(COMMON_FEATURES, medians)
        ]
    )


def fit_source_processor(frame, relative_error=0.001):
    if not 0 < relative_error <= 0.1:
        raise ValueError(
            "relative_error must be in (0, 0.1]; zero uses expensive exact quantiles"
        )
    source = scalar_features(frame)
    if not source.limit(1).count():
        raise ValueError("Empty source train")
    quantiles = source.approxQuantile(list(COMMON_FEATURES), [0.5], relative_error)
    medians = tuple(values[0] if values else 0.0 for values in quantiles)
    logged = log_imputed(source, medians)
    quantiles = logged.approxQuantile(
        list(COMMON_FEATURES), [0.25, 0.5, 0.75], relative_error
    )
    centers = tuple(values[1] for values in quantiles)
    scales = tuple(
        values[2] - values[0] if values[2] - values[0] > 1e-12 else 1.0
        for values in quantiles
    )
    if not all(math.isfinite(value) for value in (*medians, *centers, *scales)):
        raise ValueError("Nonfinite fitted statistics")
    return DistributedSourceProcessor(
        medians,
        centers,
        scales,
        relative_error,
        frame.sparkSession.version,
        schema_hash(),
    )


def transform(frame, processor):
    require_columns(frame, ["label"])
    features = scalar_features(frame)
    if "features" in frame.columns and not set(COMMON_FEATURES) <= set(frame.columns):
        values = [F.col("features")[i].cast("double") for i in range(5)]
    else:
        values = [F.col(name).cast("double") for name in COMMON_FEATURES]
    output = []
    for value, median, center, scale in zip(
        values, processor.medians, processor.centers, processor.scales
    ):
        imputed = F.when(finite(value), value).otherwise(F.lit(median))
        logged = F.signum(imputed) * F.log1p(F.abs(imputed))
        output.append(((logged - center) / scale).cast("float"))
    return frame.select(
        F.array(*output).alias("features"), F.col("label").cast("long").alias("label")
    )


def write_prepared(frame, destination, processor):
    from functools import reduce

    prepared = transform(frame, processor)
    invalid = reduce(
        lambda a, b: a | b, [~finite(F.col("features")[i]) for i in range(5)]
    )
    invalid_label = F.col("label").isNull() | ~F.col("label").isin(0, 1)
    if prepared.filter(invalid | invalid_label).limit(1).count():
        raise ValueError("Invalid prepared features or labels")
    write_parquet(prepared, destination)
