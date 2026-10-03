from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from scipy.stats import ks_2samp, spearmanr, wasserstein_distance
from sklearn.feature_selection import mutual_info_classif

from features.common_features import (
    canonicalize_frame,
    load_common_feature_config,
)

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CONFIG = ROOT / "configs" / "common_features_v2.json"

DEFAULT_SPLIT_ROOT = ROOT / "data" / "splits"

DEFAULT_OUTPUT_ROOT = ROOT / "results" / "feature_audit"

BATCH_SIZE = 65_536
DEFAULT_SAMPLE_ROWS = 200_000
RANDOM_STATE = 42


# ============================================================
# Helpers
# ============================================================


def parquet_files(directory: Path):
    files = sorted(directory.glob("*.parquet"))

    if not files:
        raise FileNotFoundError(f"No parquet files found in: {directory}")

    return files


def source_columns(config, domain):
    return [item[domain] for item in config]


def read_canonical_batches(
    split_root: Path,
    domain: str,
    config,
):
    """
    Stream one canonicalized batch at a time.

    Important:
    - unit conversion happens inside canonicalize_frame()
    - no imputation
    - NaN remains NaN
    """

    directory = split_root / f"{domain}_train"

    raw_columns = source_columns(config, domain) + ["label"]

    for path in parquet_files(directory):

        with pq.ParquetFile(path) as parquet:

            for batch in parquet.iter_batches(
                batch_size=BATCH_SIZE,
                columns=raw_columns,
            ):

                raw = batch.to_pandas()

                canonical = canonicalize_frame(
                    raw,
                    domain,
                    config,
                )

                yield canonical


# ============================================================
# 1. Exact data-quality audit
# ============================================================


def audit_quality(
    split_root: Path,
    domain: str,
    config,
):
    """
    Exact streaming statistics over ALL train rows.
    """

    feature_names = [item["canonical_name"] for item in config]

    total_rows = 0

    stats = {
        feature: {
            "finite": 0,
            "missing": 0,
            "zero": 0,
        }
        for feature in feature_names
    }

    for frame in read_canonical_batches(
        split_root,
        domain,
        config,
    ):

        total_rows += len(frame)

        for feature in feature_names:

            values = frame[feature].to_numpy(
                dtype=np.float64,
                copy=False,
            )

            finite_mask = np.isfinite(values)

            finite_values = values[finite_mask]

            stats[feature]["finite"] += int(finite_mask.sum())

            stats[feature]["missing"] += int((~finite_mask).sum())

            stats[feature]["zero"] += int(np.sum(finite_values == 0))

    rows = []

    for feature in feature_names:

        finite = stats[feature]["finite"]
        missing = stats[feature]["missing"]
        zero = stats[feature]["zero"]

        rows.append(
            {
                "domain": domain,
                "feature": feature,
                "rows": total_rows,
                "finite_count": finite,
                "missing_count": missing,
                "finite_rate": (finite / total_rows if total_rows else np.nan),
                "missing_rate": (missing / total_rows if total_rows else np.nan),
                "zero_rate_among_finite": (zero / finite if finite else np.nan),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# 2. Deterministic sample for expensive statistics
# ============================================================


def sample_domain(
    split_root: Path,
    domain: str,
    config,
    max_rows: int,
    seed: int,
):
    """
    Create a deterministic random sample without loading
    the entire dataset into RAM.

    Each batch receives a random key; final sample is drawn
    from the accumulated capped pool.
    """

    rng = np.random.default_rng(seed)

    feature_names = [item["canonical_name"] for item in config]

    collected = []
    collected_rows = 0

    # Slightly over-collect then sample down.
    pool_limit = max_rows * 2

    for frame in read_canonical_batches(
        split_root,
        domain,
        config,
    ):

        if collected_rows >= pool_limit:
            break

        remaining = pool_limit - collected_rows

        if len(frame) > remaining:

            idx = rng.choice(
                len(frame),
                size=remaining,
                replace=False,
            )

            frame = frame.iloc[idx]

        collected.append(frame[feature_names + ["label"]].copy())

        collected_rows += len(frame)

    if not collected:
        raise RuntimeError(f"No rows sampled for {domain}")

    pool = pd.concat(
        collected,
        ignore_index=True,
    )

    if len(pool) > max_rows:

        indices = rng.choice(
            len(pool),
            size=max_rows,
            replace=False,
        )

        pool = pool.iloc[indices].reset_index(drop=True)

    return pool


# ============================================================
# 3. Mutual Information
# ============================================================


def compute_mutual_information(
    frame: pd.DataFrame,
    feature_names,
    seed: int,
):
    """
    MI(feature ; label).

    Missing values are median-imputed ONLY for the MI
    calculation.

    This is a diagnostic, not part of preprocessing.
    """

    rows = []

    y = frame["label"].to_numpy(dtype=np.int64)

    for feature in feature_names:

        x = frame[[feature]].copy()

        finite = np.isfinite(x[feature].to_numpy())

        if finite.sum() == 0:

            mi = np.nan

        else:

            median = np.nanmedian(x[feature].to_numpy(dtype=np.float64))

            x[feature] = (
                x[feature]
                .replace(
                    [np.inf, -np.inf],
                    np.nan,
                )
                .fillna(median)
            )

            mi = mutual_info_classif(
                x.to_numpy(),
                y,
                discrete_features=False,
                random_state=seed,
            )[0]

        rows.append(
            {
                "feature": feature,
                "mutual_information": mi,
            }
        )

    result = pd.DataFrame(rows)

    max_mi = result["mutual_information"].max()

    if pd.notna(max_mi) and max_mi > 0:

        result["mi_normalized"] = result["mutual_information"] / max_mi

    else:

        result["mi_normalized"] = np.nan

    return result


# ============================================================
# 4. Spearman redundancy
# ============================================================


def compute_redundancy(
    frame: pd.DataFrame,
    feature_names,
):
    """
    Pairwise Spearman correlation.

    No imputation:
    pairwise finite observations only.
    """

    matrix = pd.DataFrame(
        np.nan,
        index=feature_names,
        columns=feature_names,
        dtype=float,
    )

    for f1 in feature_names:

        for f2 in feature_names:

            x = frame[f1].to_numpy(dtype=np.float64)

            y = frame[f2].to_numpy(dtype=np.float64)

            mask = np.isfinite(x) & np.isfinite(y)

            if mask.sum() < 3:
                rho = np.nan

            elif f1 == f2:
                rho = 1.0

            else:

                rho, _ = spearmanr(
                    x[mask],
                    y[mask],
                )

            matrix.loc[f1, f2] = rho

    # Feature-level summary:
    # strongest correlation with another feature.
    rows = []

    for feature in feature_names:

        correlations = matrix.loc[feature].drop(feature).abs()

        if correlations.notna().any():

            strongest_feature = correlations.idxmax()

            max_rho = correlations.max()

        else:

            strongest_feature = None
            max_rho = np.nan

        rows.append(
            {
                "feature": feature,
                "max_abs_spearman": max_rho,
                "most_correlated_with": (strongest_feature),
                "redundancy_flag_095": (
                    bool(max_rho >= 0.95) if pd.notna(max_rho) else False
                ),
            }
        )

    summary = pd.DataFrame(rows)

    return matrix, summary


# ============================================================
# 5. Domain-shift diagnostic
# ============================================================


def robust_standardize(
    source_values,
    target_values,
):
    """
    Fit robust location/scale using SOURCE ONLY.

    This makes Wasserstein distances more comparable
    across features.
    """

    source_values = np.asarray(
        source_values,
        dtype=np.float64,
    )

    target_values = np.asarray(
        target_values,
        dtype=np.float64,
    )

    median = np.median(source_values)

    q25, q75 = np.percentile(
        source_values,
        [25, 75],
    )

    iqr = q75 - q25

    if not np.isfinite(iqr) or iqr <= 0:
        iqr = 1.0

    source_scaled = (source_values - median) / iqr

    target_scaled = (target_values - median) / iqr

    return (
        source_scaled,
        target_scaled,
        median,
        iqr,
    )


def compute_domain_shift(
    source_frame,
    target_frame,
    feature_names,
):
    """
    Diagnostic only.

    KS:
      shape/location distribution difference.

    Wasserstein:
      distance after SOURCE-fitted robust scaling.

    Do NOT use these metrics as semantic KEEP/DROP rules.
    """

    rows = []

    for feature in feature_names:

        source = source_frame[feature].to_numpy(dtype=np.float64)

        target = target_frame[feature].to_numpy(dtype=np.float64)

        source = source[np.isfinite(source)]

        target = target[np.isfinite(target)]

        if len(source) < 2 or len(target) < 2:

            rows.append(
                {
                    "feature": feature,
                    "ks_statistic": np.nan,
                    "ks_pvalue": np.nan,
                    "wasserstein_robust": np.nan,
                    "source_median": np.nan,
                    "source_iqr": np.nan,
                }
            )

            continue

        ks = ks_2samp(
            source,
            target,
            alternative="two-sided",
            method="auto",
        )

        (
            source_scaled,
            target_scaled,
            source_median,
            source_iqr,
        ) = robust_standardize(
            source,
            target,
        )

        wd = wasserstein_distance(
            source_scaled,
            target_scaled,
        )

        rows.append(
            {
                "feature": feature,
                "ks_statistic": (float(ks.statistic)),
                "ks_pvalue": (float(ks.pvalue)),
                "wasserstein_robust": (float(wd)),
                "source_median": (float(source_median)),
                "source_iqr": (float(source_iqr)),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# Main direction audit
# ============================================================


def audit_direction(
    direction,
    source,
    target,
    source_sample,
    target_sample,
    quality_source,
    quality_target,
    feature_names,
    output_root,
    seed,
):
    # --------------------------------------------------------
    # MI: SOURCE LABEL ONLY
    # --------------------------------------------------------

    mi = compute_mutual_information(
        source_sample,
        feature_names,
        seed,
    )

    mi = mi.rename(
        columns={
            "mutual_information": "source_mutual_information",
            "mi_normalized": "source_mi_normalized",
        }
    )

    # --------------------------------------------------------
    # Redundancy: SOURCE only
    # --------------------------------------------------------

    (
        corr_matrix,
        redundancy,
    ) = compute_redundancy(
        source_sample,
        feature_names,
    )

    # --------------------------------------------------------
    # Domain shift: source vs unlabeled target
    # --------------------------------------------------------

    shift = compute_domain_shift(
        source_sample,
        target_sample,
        feature_names,
    )

    # --------------------------------------------------------
    # Quality merge
    # --------------------------------------------------------

    qs = quality_source.drop(columns=["domain"]).rename(
        columns={
            col: f"source_{col}"
            for col in quality_source.columns
            if col
            not in {
                "domain",
                "feature",
            }
        }
    )

    qt = quality_target.drop(columns=["domain"]).rename(
        columns={
            col: f"target_{col}"
            for col in quality_target.columns
            if col
            not in {
                "domain",
                "feature",
            }
        }
    )

    result = (
        qs.merge(
            qt,
            on="feature",
            validate="one_to_one",
        )
        .merge(
            mi,
            on="feature",
            validate="one_to_one",
        )
        .merge(
            redundancy,
            on="feature",
            validate="one_to_one",
        )
        .merge(
            shift,
            on="feature",
            validate="one_to_one",
        )
    )

    result["min_finite_rate"] = result[
        [
            "source_finite_rate",
            "target_finite_rate",
        ]
    ].min(axis=1)

    # --------------------------------------------------------
    # IMPORTANT:
    # This is a REVIEW flag, not automatic feature selection.
    # --------------------------------------------------------

    result["quality_flag_098"] = result["min_finite_rate"] < 0.98

    result["low_source_information"] = result["source_mutual_information"] <= 0

    result = result.sort_values(
        "source_mutual_information",
        ascending=False,
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = output_root / f"candidate_stats_{direction}.csv"

    corr_path = output_root / f"spearman_{direction}.csv"

    result.to_csv(
        summary_path,
        index=False,
    )

    corr_matrix.to_csv(corr_path)

    print(f"\n[{direction}]")

    print(
        result[
            [
                "feature",
                "min_finite_rate",
                "source_mutual_information",
                "max_abs_spearman",
                "ks_statistic",
                "wasserstein_robust",
            ]
        ].to_string(index=False)
    )

    print(f"\nSaved: {summary_path}")

    print(f"Saved: {corr_path}")


# ============================================================
# Entry point
# ============================================================


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
    )

    parser.add_argument(
        "--split-root",
        type=Path,
        default=DEFAULT_SPLIT_ROOT,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )

    parser.add_argument(
        "--sample-rows",
        type=int,
        default=DEFAULT_SAMPLE_ROWS,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_STATE,
    )

    args = parser.parse_args()

    config = load_common_feature_config(args.config)

    feature_names = [item["canonical_name"] for item in config]

    print(f"Features ({len(feature_names)}):")

    print(", ".join(feature_names))

    # --------------------------------------------------------
    # Exact quality: ALL training rows
    # --------------------------------------------------------

    print("\nComputing exact UNSW quality...")

    quality_unsw = audit_quality(
        args.split_root,
        "unsw",
        config,
    )

    print("Computing exact CICIDS quality...")

    quality_cicids = audit_quality(
        args.split_root,
        "cicids",
        config,
    )

    quality_all = pd.concat(
        [
            quality_unsw,
            quality_cicids,
        ],
        ignore_index=True,
    )

    args.output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    quality_path = args.output_root / "feature_quality.csv"

    quality_all.to_csv(
        quality_path,
        index=False,
    )

    print(f"Saved: {quality_path}")

    # --------------------------------------------------------
    # Samples for expensive diagnostics
    # --------------------------------------------------------

    print(f"\nSampling up to " f"{args.sample_rows:,} UNSW rows...")

    unsw_sample = sample_domain(
        args.split_root,
        "unsw",
        config,
        args.sample_rows,
        args.seed,
    )

    print(f"UNSW sample rows: " f"{len(unsw_sample):,}")

    print(f"\nSampling up to " f"{args.sample_rows:,} CICIDS rows...")

    cicids_sample = sample_domain(
        args.split_root,
        "cicids",
        config,
        args.sample_rows,
        args.seed + 1,
    )

    print(f"CICIDS sample rows: " f"{len(cicids_sample):,}")

    # --------------------------------------------------------
    # Direction 1
    # --------------------------------------------------------

    audit_direction(
        direction="unsw_to_cicids",
        source="unsw",
        target="cicids",
        source_sample=unsw_sample,
        target_sample=cicids_sample,
        quality_source=quality_unsw,
        quality_target=quality_cicids,
        feature_names=feature_names,
        output_root=args.output_root,
        seed=args.seed,
    )

    # --------------------------------------------------------
    # Direction 2
    # --------------------------------------------------------

    audit_direction(
        direction="cicids_to_unsw",
        source="cicids",
        target="unsw",
        source_sample=cicids_sample,
        target_sample=unsw_sample,
        quality_source=quality_cicids,
        quality_target=quality_unsw,
        feature_names=feature_names,
        output_root=args.output_root,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
