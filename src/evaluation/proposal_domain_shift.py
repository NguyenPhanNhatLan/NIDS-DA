from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
)

from features.common_features import COMMON_FEATURES
from training.adaptation import mmd_loss


ROOT = Path(__file__).resolve().parents[2]

FEATURE_ROOT = (
    ROOT
    / "data"
    / "features"
    / "proposal_v1"
)

RESULT_ROOT = (
    ROOT
    / "results"
    / "proposal_v1"
    / "domain_shift"
)

COMMON_CONFIG = (
    ROOT
    / "configs"
    / "common_features_v1.json"
)


# =========================================================
# Basic helpers
# =========================================================

def direction_domains(direction):
    if direction == "unsw_to_cicids":
        return "unsw", "cicids"

    if direction == "cicids_to_unsw":
        return "cicids", "unsw"

    raise ValueError(
        f"Unknown direction: {direction}"
    )


def sha256(path):
    path = Path(path)

    h = hashlib.sha256()

    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def parquet_files(directory):
    files = sorted(
        Path(directory).glob("*.parquet")
    )

    if not files:
        raise FileNotFoundError(
            f"No parquet files in {directory}"
        )

    return files


# =========================================================
# Uniform feature sampling
#
# Important:
# - reads ONLY "features"
# - never reads intrusion labels
# - avoids loading 2M rows into RAM
# =========================================================

def sample_features(
    directory,
    max_samples,
    seed,
):
    files = parquet_files(directory)

    row_counts = []

    for path in files:
        parquet = pq.ParquetFile(path)

        row_counts.append(
            parquet.metadata.num_rows
        )

    total_rows = int(
        sum(row_counts)
    )

    if total_rows == 0:
        raise ValueError(
            f"Empty dataset: {directory}"
        )

    sample_size = min(
        int(max_samples),
        total_rows,
    )

    rng = np.random.default_rng(seed)

    # Uniform random global row indices
    selected = np.sort(
        rng.choice(
            total_rows,
            size=sample_size,
            replace=False,
        )
    )

    output = []

    global_offset = 0

    for path in files:

        with pq.ParquetFile(path) as parquet:

            for batch in parquet.iter_batches(
                batch_size=65536,
                columns=["features"],
            ):

                features = batch.column(
                    "features"
                )

                n_rows = len(features)

                batch_start = global_offset
                batch_end = (
                    global_offset + n_rows
                )

                left = np.searchsorted(
                    selected,
                    batch_start,
                    side="left",
                )

                right = np.searchsorted(
                    selected,
                    batch_end,
                    side="left",
                )

                if right > left:

                    local_indices = (
                        selected[left:right]
                        - batch_start
                    )

                    # FixedSizeListArray -> matrix
                    values = (
                        features
                        .values
                        .to_numpy(
                            zero_copy_only=False
                        )
                    )

                    input_dim = (
                        features.type.list_size
                    )

                    matrix = values.reshape(
                        n_rows,
                        input_dim,
                    )

                    output.append(
                        matrix[
                            local_indices
                        ].astype(
                            np.float32,
                            copy=True,
                        )
                    )

                global_offset = batch_end

    if not output:
        raise RuntimeError(
            "Sampling produced no rows"
        )

    result = np.concatenate(
        output,
        axis=0,
    )

    if result.shape[0] != sample_size:
        raise RuntimeError(
            "Sample row count mismatch: "
            f"expected={sample_size}, "
            f"actual={len(result)}"
        )

    if result.ndim != 2:
        raise ValueError(
            f"Expected matrix, got "
            f"{result.shape}"
        )

    if result.shape[1] != len(
        COMMON_FEATURES
    ):
        raise ValueError(
            f"Expected "
            f"{len(COMMON_FEATURES)} "
            f"features, got "
            f"{result.shape[1]}"
        )

    if not np.isfinite(result).all():
        raise ValueError(
            f"NaN/Inf found in {directory}"
        )

    return result, total_rows


# =========================================================
# 1. KS test
# =========================================================

def compute_ks(
    source,
    target,
):
    results = {}

    for index, feature in enumerate(
        COMMON_FEATURES
    ):

        source_values = (
            source[:, index]
        )

        target_values = (
            target[:, index]
        )

        test = ks_2samp(
            source_values,
            target_values,
            alternative="two-sided",
            method="auto",
        )

        results[feature] = {
            "ks_statistic": float(
                test.statistic
            ),
            "p_value": float(
                test.pvalue
            ),
            "source_median": float(
                np.median(
                    source_values
                )
            ),
            "target_median": float(
                np.median(
                    target_values
                )
            ),
            "source_mean": float(
                np.mean(
                    source_values
                )
            ),
            "target_mean": float(
                np.mean(
                    target_values
                )
            ),
        }

    return results


# =========================================================
# 2. Input-space MMD
#
# Uses EXACTLY the existing project MMD:
# - single RBF
# - median heuristic
# - biased empirical MMD²
#
# Repeat subsampling to reduce sampling noise.
# =========================================================

def compute_input_mmd(
    source,
    target,
    sample_size=2048,
    repeats=5,
    seed=42,
):
    n = min(
        int(sample_size),
        len(source),
        len(target),
    )

    if n < 2:
        raise ValueError(
            "MMD requires at least 2 rows"
        )

    rng = np.random.default_rng(seed)

    mmd_values = []
    bandwidth_values = []

    for repeat in range(repeats):

        source_index = rng.choice(
            len(source),
            size=n,
            replace=False,
        )

        target_index = rng.choice(
            len(target),
            size=n,
            replace=False,
        )

        source_tensor = (
            torch.from_numpy(
                source[source_index]
            )
            .float()
        )

        target_tensor = (
            torch.from_numpy(
                target[target_index]
            )
            .float()
        )

        # CPU intentionally:
        # deterministic diagnostic,
        # no gradient required.
        with torch.no_grad():

            loss, bandwidth = (
                mmd_loss(
                    source_tensor,
                    target_tensor,
                )
            )

        mmd_values.append(
            float(loss.item())
        )

        bandwidth_values.append(
            float(bandwidth.item())
        )

    mmd_array = np.asarray(
        mmd_values,
        dtype=np.float64,
    )

    bandwidth_array = np.asarray(
        bandwidth_values,
        dtype=np.float64,
    )

    return {
        "sample_size_per_domain": n,
        "repeats": int(repeats),

        "mmd2_mean": float(
            mmd_array.mean()
        ),

        "mmd2_std": float(
            mmd_array.std(ddof=1)
            if repeats > 1
            else 0.0
        ),

        "mmd2_runs": [
            float(value)
            for value in mmd_values
        ],

        "bandwidth_mean": float(
            bandwidth_array.mean()
        ),

        "bandwidth_std": float(
            bandwidth_array.std(ddof=1)
            if repeats > 1
            else 0.0
        ),
    }


# =========================================================
# 3. Domain-classifier AUC
#
# domain=0 -> source
# domain=1 -> target
#
# No intrusion labels.
#
# AUC ~ 0.5:
#   domains hard to distinguish
#
# AUC -> 1:
#   strong domain shift
# =========================================================

def compute_domain_auc(
    source,
    target,
    seed=42,
    folds=5,
):
    n = min(
        len(source),
        len(target),
    )

    source = source[:n]
    target = target[:n]

    x = np.concatenate(
        [
            source,
            target,
        ],
        axis=0,
    )

    y = np.concatenate(
        [
            np.zeros(
                n,
                dtype=np.int64,
            ),
            np.ones(
                n,
                dtype=np.int64,
            ),
        ],
        axis=0,
    )

    model = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        random_state=seed,
    )

    cv = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=seed,
    )

    scores = cross_val_score(
        model,
        x,
        y,
        scoring="roc_auc",
        cv=cv,
    )

    return {
        "classifier":
            "logistic_regression",

        "samples_per_domain":
            int(n),

        "cv_folds":
            int(folds),

        "auc_mean":
            float(scores.mean()),

        "auc_std":
            float(
                scores.std(ddof=1)
                if len(scores) > 1
                else 0.0
            ),

        "auc_folds": [
            float(score)
            for score in scores
        ],
    }


# =========================================================
# Run one direction
# =========================================================

def run(
    direction,
    seed=42,
    analysis_sample=100_000,
    mmd_sample=2048,
    mmd_repeats=5,
    cv_folds=5,
):
    source_domain, target_domain = (
        direction_domains(direction)
    )

    base = (
        FEATURE_ROOT
        / direction
    )

    source_path = (
        base
        / f"{source_domain}_train"
    )

    target_path = (
        base
        / f"{target_domain}_train"
    )

    print(
        f"\n=== {direction} ==="
    )

    print(
        f"Source train: {source_path}"
    )

    print(
        f"Target train: {target_path}"
    )

    # Different seed offsets avoid
    # selecting identical index patterns
    # accidentally across domains.
    source, source_total = (
        sample_features(
            source_path,
            analysis_sample,
            seed,
        )
    )

    target, target_total = (
        sample_features(
            target_path,
            analysis_sample,
            seed + 1,
        )
    )

    print(
        f"Source rows: "
        f"{source_total:,} "
        f"(sample={len(source):,})"
    )

    print(
        f"Target rows: "
        f"{target_total:,} "
        f"(sample={len(target):,})"
    )

    # ---------------------------------
    # KS
    # ---------------------------------

    ks = compute_ks(
        source,
        target,
    )

    # ---------------------------------
    # MMD
    # ---------------------------------

    mmd = compute_input_mmd(
        source,
        target,
        sample_size=mmd_sample,
        repeats=mmd_repeats,
        seed=seed,
    )

    # ---------------------------------
    # Domain classifier
    # ---------------------------------

    domain_auc = compute_domain_auc(
        source,
        target,
        seed=seed,
        folds=cv_folds,
    )

    # ---------------------------------
    # Provenance
    # ---------------------------------

    preprocessor_path = (
        ROOT
        / "models"
        / "proposal_v1"
        / direction
        / "preprocessor.joblib"
    )

    if not preprocessor_path.exists():
        raise FileNotFoundError(
            f"Missing fitted processor: "
            f"{preprocessor_path}"
        )

    result = {
        "protocol":
            "proposal_suite_v1",

        "analysis":
            "pre_adaptation_domain_shift",

        "direction":
            direction,

        "source_domain":
            source_domain,

        "target_domain":
            target_domain,

        "seed":
            seed,

        "feature_count":
            len(COMMON_FEATURES),

        "features":
            list(COMMON_FEATURES),

        "input_space":
            (
                "10D common feature space "
                "after source-train fitted "
                "median -> signed_log1p "
                "-> RobustScaler"
            ),

        "data_roles": {
            "source":
                f"{source_domain}_train",

            "target":
                f"{target_domain}_train",

            "target_intrusion_labels_used":
                False,
        },

        "sampling": {
            "analysis_sample":
                int(analysis_sample),

            "source_total_rows":
                int(source_total),

            "target_total_rows":
                int(target_total),

            "source_sample_rows":
                int(len(source)),

            "target_sample_rows":
                int(len(target)),
        },

        "provenance": {
            "common_feature_config":
                str(COMMON_CONFIG),

            "common_feature_config_sha256":
                sha256(COMMON_CONFIG),

            "preprocessor":
                str(preprocessor_path),

            "preprocessor_sha256":
                sha256(
                    preprocessor_path
                ),
        },

        "ks_by_feature":
            ks,

        "input_mmd":
            mmd,

        "domain_classifier":
            domain_auc,
    }

    # ---------------------------------
    # Save
    # ---------------------------------

    RESULT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = (
        RESULT_ROOT
        / f"{direction}_seed{seed}.json"
    )

    if output.exists():
        raise FileExistsError(
            f"Result already exists: "
            f"{output}"
        )

    with output.open(
        "x",
        encoding="utf-8",
    ) as stream:

        json.dump(
            result,
            stream,
            indent=2,
            allow_nan=False,
        )

        stream.write("\n")

    # ---------------------------------
    # Console summary
    # ---------------------------------

    print(
        "\nInput MMD²:"
    )

    print(
        f"{mmd['mmd2_mean']:.6f} "
        f"± "
        f"{mmd['mmd2_std']:.6f}"
    )

    print(
        "\nDomain classifier AUC:"
    )

    print(
        f"{domain_auc['auc_mean']:.6f} "
        f"± "
        f"{domain_auc['auc_std']:.6f}"
    )

    print(
        "\nKS statistics:"
    )

    sorted_ks = sorted(
        ks.items(),
        key=lambda item:
            item[1]["ks_statistic"],
        reverse=True,
    )

    for feature, values in sorted_ks:

        print(
            f"{feature:24s} "
            f"KS="
            f"{values['ks_statistic']:.6f} "
            f"p="
            f"{values['p_value']:.3e}"
        )

    print(
        f"\nSaved: {output}"
    )

    return result


# =========================================================
# CLI
# =========================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Pre-adaptation domain shift "
            "diagnostics for proposal suite."
        )
    )

    parser.add_argument(
        "--direction",
        required=True,
        choices=[
            "unsw_to_cicids",
            "cicids_to_unsw",
            "both",
        ],
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--analysis-sample",
        type=int,
        default=100_000,
        help=(
            "Maximum rows sampled per "
            "domain for KS/domain AUC."
        ),
    )

    parser.add_argument(
        "--mmd-sample",
        type=int,
        default=2048,
    )

    parser.add_argument(
        "--mmd-repeats",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    if args.analysis_sample < 2:
        parser.error(
            "analysis-sample must be >= 2"
        )

    if args.mmd_sample < 2:
        parser.error(
            "mmd-sample must be >= 2"
        )

    if args.mmd_repeats < 1:
        parser.error(
            "mmd-repeats must be >= 1"
        )

    if args.cv_folds < 2:
        parser.error(
            "cv-folds must be >= 2"
        )

    if args.direction == "both":

        for direction in (
            "unsw_to_cicids",
            "cicids_to_unsw",
        ):
            run(
                direction,
                seed=args.seed,
                analysis_sample=
                    args.analysis_sample,
                mmd_sample=
                    args.mmd_sample,
                mmd_repeats=
                    args.mmd_repeats,
                cv_folds=
                    args.cv_folds,
            )

    else:

        run(
            args.direction,
            seed=args.seed,
            analysis_sample=
                args.analysis_sample,
            mmd_sample=
                args.mmd_sample,
            mmd_repeats=
                args.mmd_repeats,
            cv_folds=
                args.cv_folds,
        )


if __name__ == "__main__":
    main()