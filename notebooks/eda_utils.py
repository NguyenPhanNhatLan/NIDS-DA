"""Read-only EDA helpers shared by UNSW and CICIDS notebooks."""

import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import functions as F

NUMERIC_TYPES = {
    "float",
    "integer",
    "integer/float",
    "integer/string",
    "binary",
    "timestamp",
}
TARGETS = {"label", "binary_label", "raw_label", "attack_cat"}
NO_LOG = {
    "sport",
    "dsport",
    "source_port",
    "destination_port",
    "protocol",
    "is_ftp_login",
    "is_sm_ips_ports",
}
IDENTIFIERS = {
    "srcip",
    "dstip",
    "sport",
    "dsport",
    "source_port",
    "destination_port",
    "id",
    "flow_id",
    "timestamp",
    "stime",
    "ltime",
    "stcpb",
    "dtcpb",
}


def dictionary_types(root, domain):
    path = Path(root) / "docs/data_dictionary" / f"feature_dictionary_{domain}.csv"
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return {row["feature"]: row["feature_type"] for row in csv.DictReader(stream)}


def type_for(name, types):
    name = "ct_src_ltm" if name == "ct_src__ltm" else name
    return types.get(name, types.get(re.sub(r"_dup\d+$", "", name)))


def numeric_names(frame, types):
    return [
        name
        for name in frame.columns
        if type_for(name, types) in NUMERIC_TYPES and name not in TARGETS
    ]


def finite(value):
    return value.isNotNull() & ~F.isnan(value) & (F.abs(value) != float("inf"))


def quality_table(frame, types):
    expressions = [F.count("*").alias("rows")]
    for i, name in enumerate(frame.columns):
        text = F.trim(F.col(name).cast("string"))
        missing = (
            text.isNull() | (text == "") | F.upper(text).isin("NULL", "NONE", "N/A")
        )
        expressions.append(F.sum(F.when(missing, 1).otherwise(0)).alias(f"missing_{i}"))
        if type_for(name, types) in NUMERIC_TYPES:
            value = text.cast("double")
            expressions.extend(
                [
                    F.sum(F.when(F.isnan(value), 1).otherwise(0)).alias(f"nan_{i}"),
                    F.sum(F.when(F.abs(value) == float("inf"), 1).otherwise(0)).alias(
                        f"inf_{i}"
                    ),
                    F.sum(F.when(~missing & value.isNull(), 1).otherwise(0)).alias(
                        f"parse_{i}"
                    ),
                ]
            )
    summary = frame.agg(*expressions).first().asDict()
    rows = []
    for i, name in enumerate(frame.columns):
        counts = {
            key: int(summary.get(f"{key}_{i}") or 0)
            for key in ("missing", "nan", "inf", "parse")
        }
        rows.append(
            {
                "column": name,
                "declared_type": type_for(name, types) or "unmapped",
                **counts,
                "missing_pct": (
                    100 * counts["missing"] / summary["rows"] if summary["rows"] else 0
                ),
            }
        )
    table = pd.DataFrame(rows)
    table["issues"] = table[["missing", "nan", "inf", "parse"]].sum(axis=1)
    return (
        table.sort_values(["issues", "column"], ascending=[False, True]),
        summary["rows"],
    )


def clean_for_eda(raw, domain, types):
    expressions = []
    for name in raw.columns:
        text = F.trim(F.col(name).cast("string"))
        missing = (
            text.isNull()
            | (text == "")
            | F.upper(text).isin("NULL", "NONE", "N/A", "NAN")
        )
        text = F.when(missing, F.lit(None)).otherwise(text)
        if name not in TARGETS and type_for(name, types) in NUMERIC_TYPES:
            value = text.cast("double")
            if domain == "unsw" and name == "ct_ftp_cmd":
                value = F.coalesce(value, F.lit(0.0))
            expression = F.when(finite(value), value)
        else:
            expression = text
        expressions.append(expression.alias(name))
    frame = raw.select(*expressions)
    if domain == "unsw":
        value = F.col("label").cast("double")
        frame = frame.withColumn(
            "label", F.when(value.isin(0.0, 1.0), value.cast("long"))
        )
        if "ct_flw_http_mthd" in frame.columns:
            frame = frame.fillna(0.0, subset=["ct_flw_http_mthd"])
        if "is_ftp_login" in frame.columns:
            value = F.coalesce(F.col("is_ftp_login"), F.lit(0.0))
            frame = frame.withColumn(
                "is_ftp_login", F.when(value > 1, 1.0).when(value.isin(0.0, 1.0), value)
            )
        if "service" in frame.columns:
            frame = frame.withColumn(
                "service",
                F.when(
                    F.col("service").isNull() | (F.col("service") == "-"), "unknown"
                ).otherwise(F.col("service")),
            )
        if "attack_cat" in frame.columns:
            frame = frame.withColumn(
                "attack_cat",
                F.when(F.col("label") == 0, "Normal")
                .when(F.col("label").isNull(), "Unknown-Label")
                .when(F.col("attack_cat").isNull(), "Unknown-Attack")
                .otherwise(F.col("attack_cat")),
            )
    else:
        text = F.col("label")
        frame = frame.withColumn(
            "binary_label",
            F.when(F.upper(text) == "BENIGN", 0)
            .when(text.isNotNull() & ~text.rlike(r"^[+-]?\d+(\.\d+)?$"), 1)
            .cast("long"),
        )
    return frame


def numeric_statistics(frame, columns):
    fields = ["min", "max", "mean", "std", "skewness", "approx_unique"]
    if not columns:
        return pd.DataFrame(
            columns=["feature", *fields, "abs_skewness", "suggest_log1p"]
        )
    expressions = []
    for i, name in enumerate(columns):
        value = F.when(finite(F.col(name).cast("double")), F.col(name).cast("double"))
        for key, function in [
            ("min", F.min),
            ("max", F.max),
            ("mean", F.avg),
            ("std", F.stddev_samp),
            ("skewness", F.skewness),
            ("approx_unique", F.approx_count_distinct),
        ]:
            expressions.append(function(value).alias(f"{key}_{i}"))
    result = frame.agg(*expressions).first().asDict()
    rows = [
        {"feature": name, **{key: result[f"{key}_{i}"] for key in fields}}
        for i, name in enumerate(columns)
    ]
    table = pd.DataFrame(rows)
    table["abs_skewness"] = table["skewness"].abs()
    table["suggest_log1p"] = (
        table["skewness"].gt(1)
        & table["min"].ge(0)
        & table["approx_unique"].gt(2)
        & ~table["feature"].isin(NO_LOG)
    )
    return table.sort_values("abs_skewness", ascending=False, na_position="last")


def bounded_sample(frame, columns, total_rows, max_rows=20_000, seed=42):
    if max_rows < 1:
        raise ValueError("max_rows must be positive")
    if not columns or total_rows == 0:
        return pd.DataFrame(columns=columns)
    fraction = min(1.0, 1.2 * max_rows / total_rows)
    # Bernoulli sample with a hard limit BEFORE collection. This is an EDA
    # sample, not an exact uniform sample for inferential statistics.
    return (
        frame.select(*columns)
        .sample(False, fraction, seed)
        .limit(max_rows)
        .toPandas()
        .replace([np.inf, -np.inf], np.nan)
    )


def category_counts(frame, column, top_n=15):
    """Full-data grouped counts, capped before driver collection."""
    return (
        frame.groupBy(column)
        .count()
        .orderBy(F.desc("count"), F.asc_nulls_last(column))
        .limit(top_n)
        .toPandas()
    )


def plot_categories(frame, columns, top_n=15):
    for name in columns:
        if name not in frame.columns:
            continue
        table = category_counts(frame, name, top_n)
        table[name] = table[name].fillna("(missing)").astype(str)
        fig, ax = plt.subplots(figsize=(10, max(3, len(table) * 0.3)))
        ax.barh(table[name][::-1], table["count"][::-1])
        ax.set(title=f"{name}: top {top_n} (full-data counts)", xlabel="Rows")
        fig.tight_layout()
        plt.show()
        plt.close(fig)


def plot_distributions(sample, columns):
    for name in columns:
        values = pd.to_numeric(sample[name], errors="coerce").dropna()
        if values.empty:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(12, 3))
        sns.histplot(values, bins=40, ax=axes[0])
        sns.boxplot(x=values, ax=axes[1])
        axes[0].set_title(f"{name}: histogram (sample)")
        axes[1].set_title(f"{name}: boxplot (sample)")
        fig.tight_layout()
        plt.show()
        plt.close(fig)


def correlation_view(sample, columns, max_features=40, threshold=0.95):
    selected = [name for name in columns if name in sample][:max_features]
    if len(selected) < 2 or len(sample) < 3:
        return pd.DataFrame(columns=["feature_1", "feature_2", "correlation"])
    corr = sample[selected].corr(min_periods=3)
    undefined = corr.columns[corr.isna().all()].tolist()
    print("Undefined/constant on sample:", undefined)
    plotted = corr.drop(index=undefined, columns=undefined)
    if len(plotted) >= 2:
        size = max(8, min(22, len(plotted) * 0.45))
        fig, ax = plt.subplots(figsize=(size, size))
        sns.heatmap(
            plotted,
            mask=np.triu(np.ones(plotted.shape, dtype=bool), k=1),
            cmap="coolwarm",
            center=0,
            vmin=-1,
            vmax=1,
            ax=ax,
        )
        ax.set_title(f"Pearson correlation: {len(plotted)} features (sample)")
        fig.tight_layout()
        plt.show()
        plt.close(fig)
    pairs = [
        {"feature_1": left, "feature_2": right, "correlation": corr.loc[left, right]}
        for i, left in enumerate(selected)
        for right in selected[i + 1 :]
        if pd.notna(corr.loc[left, right]) and abs(corr.loc[left, right]) > threshold
    ]
    return pd.DataFrame(
        pairs, columns=["feature_1", "feature_2", "correlation"]
    ).sort_values("correlation", key=lambda column: column.abs(), ascending=False)


def plot_log_comparison(sample, columns):
    comparisons = []
    for name in columns:
        values = pd.to_numeric(sample[name], errors="coerce").dropna()
        if len(values) < 3 or (values < 0).any():
            continue
        logged = np.log1p(values)
        comparisons.append(
            {"feature": name, "before_skew": values.skew(), "after_skew": logged.skew()}
        )
        fig, axes = plt.subplots(1, 2, figsize=(12, 3))
        sns.histplot(values, bins=40, ax=axes[0])
        sns.histplot(logged, bins=40, ax=axes[1])
        axes[0].set_title(f"{name}: original (sample)")
        axes[1].set_title(f"{name}: log1p (same sample)")
        fig.tight_layout()
        plt.show()
        plt.close(fig)
    return pd.DataFrame(comparisons, columns=["feature", "before_skew", "after_skew"])
