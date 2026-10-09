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
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer

from features.common_features import COMMON_FEATURES
from features.parquet_vectors import vector_matrix
from training.adaptation import mmd_loss, estimate_bandwidth_squared, rbf_kernel

from training.data_revision import revision_path, verify_revision

ROOT = Path(__file__).resolve().parents[2]

FEATURE_ROOT = revision_path('feature_root', ROOT / 'data/features/proposal_v2')

RESULT_ROOT = revision_path('result_root', ROOT / 'results/proposal_v2') / 'domain_shift'

COMMON_CONFIG = ROOT / "configs" / "common_features_v2.json"


# =========================================================
# Basic helpers
# =========================================================


def direction_domains(direction):
    if direction == "unsw_to_cicids":
        return "unsw", "cicids"

    if direction == "cicids_to_unsw":
        return "cicids", "unsw"

    raise ValueError(f"Unknown direction: {direction}")


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
    files = sorted(Path(directory).glob("*.parquet"))

    if not files:
        raise FileNotFoundError(f"No parquet files in {directory}")

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
    allow_nonfinite=False,
    return_indices=False,
):
    files = parquet_files(directory)

    row_counts = []

    for path in files:
        parquet = pq.ParquetFile(path)

        row_counts.append(parquet.metadata.num_rows)

    total_rows = int(sum(row_counts))

    if total_rows == 0:
        raise ValueError(f"Empty dataset: {directory}")

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
            scalar_columns = set(COMMON_FEATURES).issubset(parquet.schema_arrow.names)
            columns = list(COMMON_FEATURES) if scalar_columns else ["features"]
            for batch in parquet.iter_batches(
                batch_size=65536,
                columns=columns,
            ):
                n_rows = batch.num_rows

                batch_start = global_offset
                batch_end = global_offset + n_rows

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

                    local_indices = selected[left:right] - batch_start

                    if scalar_columns:
                        matrix = np.column_stack([
                            batch.column(name).to_numpy(zero_copy_only=False)
                            for name in COMMON_FEATURES
                        ])
                    else:
                        matrix = vector_matrix(batch.column("features"), path,
                                               allow_nonfinite=allow_nonfinite)

                    output.append(
                        matrix[local_indices].astype(
                            np.float64 if allow_nonfinite else np.float32,
                            copy=True,
                        )
                    )

                global_offset = batch_end

    if not output:
        raise RuntimeError("Sampling produced no rows")

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
        raise ValueError(f"Expected matrix, got " f"{result.shape}")

    if result.shape[1] != len(COMMON_FEATURES):
        raise ValueError(
            f"Expected "
            f"{len(COMMON_FEATURES)} "
            f"features, got "
            f"{result.shape[1]}"
        )

    if not allow_nonfinite and not np.isfinite(result).all():
        raise ValueError(f"NaN/Inf found in {directory}")

    if return_indices:
        return result, total_rows, selected
    return result, total_rows


# =========================================================
# 1. KS test
# =========================================================


def compute_ks(
    source,
    target,
):
    results = {}

    for index, feature in enumerate(COMMON_FEATURES):

        source_values = source[:, index]

        target_values = target[:, index]

        test = ks_2samp(
            source_values,
            target_values,
            alternative="two-sided",
            method="auto",
        )

        results[feature] = {
            "ks_statistic": float(test.statistic),
            "p_value": float(test.pvalue),
            "source_median": float(np.median(source_values)),
            "target_median": float(np.median(target_values)),
            "source_mean": float(np.mean(source_values)),
            "target_mean": float(np.mean(target_values)),
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
        raise ValueError("MMD requires at least 2 rows")

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

        source_tensor = torch.from_numpy(source[source_index]).float()

        target_tensor = torch.from_numpy(target[target_index]).float()

        # CPU intentionally:
        # deterministic diagnostic,
        # no gradient required.
        with torch.no_grad():

            loss, bandwidth = mmd_loss(
                source_tensor,
                target_tensor,
            )

        mmd_values.append(float(loss.item()))

        bandwidth_values.append(float(bandwidth.item()))

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
        "mmd2_mean": float(mmd_array.mean()),
        "mmd2_std": float(mmd_array.std(ddof=1) if repeats > 1 else 0.0),
        "std_interpretation": "subsampling variability; not a 95% confidence interval",
        "mmd2_runs": [float(value) for value in mmd_values],
        "bandwidth_mean": float(bandwidth_array.mean()),
        "bandwidth_std": float(bandwidth_array.std(ddof=1) if repeats > 1 else 0.0),
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
    return domain_auc_cv(source, target, seed=seed, folds=folds)[0]


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
    revision = verify_revision()
    source_domain, target_domain = direction_domains(direction)

    base = FEATURE_ROOT / direction

    source_path = base / f"{source_domain}_train"

    target_path = base / f"{target_domain}_train"

    print(f"\n=== {direction} ===")

    print(f"Source train: {source_path}")

    print(f"Target train: {target_path}")

    # Different seed offsets avoid
    # selecting identical index patterns
    # accidentally across domains.
    source, source_total = sample_features(
        source_path,
        analysis_sample,
        seed,
    )

    target, target_total = sample_features(
        target_path,
        analysis_sample,
        seed + 1,
    )

    print(f"Source rows: " f"{source_total:,} " f"(sample={len(source):,})")

    print(f"Target rows: " f"{target_total:,} " f"(sample={len(target):,})")

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
        revision_path('model_root', ROOT / 'models/proposal_v2') / direction / 'preprocessor.joblib'
    )

    if not preprocessor_path.exists():
        raise FileNotFoundError(f"Missing fitted processor: " f"{preprocessor_path}")

    result = {
        "protocol": "proposal_v2",
        **revision,
        "analysis": "pre_adaptation_domain_shift",
        "direction": direction,
        "source_domain": source_domain,
        "target_domain": target_domain,
        "seed": seed,
        "feature_count": len(COMMON_FEATURES),
        "features": list(COMMON_FEATURES),
        "input_space": (
            "5D common feature space "
            "after source-train fitted "
            "median -> signed_log1p "
            "-> RobustScaler"
        ),
        "data_roles": {
            "source": f"{source_domain}_train",
            "target": f"{target_domain}_train",
            "target_intrusion_labels_used": False,
        },
        "sampling": {
            "analysis_sample": int(analysis_sample),
            "source_total_rows": int(source_total),
            "target_total_rows": int(target_total),
            "source_sample_rows": int(len(source)),
            "target_sample_rows": int(len(target)),
        },
        "provenance": {
            "common_feature_config": str(COMMON_CONFIG),
            "common_feature_config_sha256": sha256(COMMON_CONFIG),
            "preprocessor": str(preprocessor_path),
            "preprocessor_sha256": sha256(preprocessor_path),
        },
        "ks_by_feature": ks,
        "input_mmd": mmd,
        "domain_classifier": domain_auc,
    }

    # ---------------------------------
    # Save
    # ---------------------------------

    RESULT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = RESULT_ROOT / f"{direction}_seed{seed}.json"

    if output.exists():
        raise FileExistsError(f"Result already exists: " f"{output}")

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

    print("\nInput MMD²:")

    print(f"{mmd['mmd2_mean']:.6f} " f"± " f"{mmd['mmd2_std']:.6f}")

    print("\nDomain classifier AUC:")

    print(f"{domain_auc['auc_mean']:.6f} " f"± " f"{domain_auc['auc_std']:.6f}")

    print("\nKS statistics:")

    sorted_ks = sorted(
        ks.items(),
        key=lambda item: item[1]["ks_statistic"],
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

    print(f"\nSaved: {output}")

    return result


# =========================================================
# CLI
# =========================================================


def fixed_kernel_mmd(source, target, bandwidth_squared):
    """Evaluate one fixed kernel, including unequal sample sizes."""
    return (rbf_kernel(source, source, bandwidth_squared).mean()
            + rbf_kernel(target, target, bandwidth_squared).mean()
            - 2 * rbf_kernel(source, target, bandwidth_squared).mean())


def intrinsic_mmd(source, target, sample_size=2048, repeats=5, seed=42,
                  alternative_spaces=None):
    # Float64 and a rounding-scale tolerance, rather than sampling tolerance.
    epsilon = 64 * np.finfo(np.float64).eps
    n = min(sample_size, len(source) // 2, len(target) // 2)
    if n < 2 or repeats < 1:
        raise ValueError("Intrinsic MMD needs >=4 rows/domain and >=1 repeat")
    rng_u = np.random.default_rng(seed)
    rng_c = np.random.default_rng(seed + 1)
    records, indices, sensitivities = [], {}, {}
    with torch.no_grad():
        for repeat in range(repeats):
            ui = rng_u.choice(len(source), 2 * n, replace=False)
            ci = rng_c.choice(len(target), 2 * n, replace=False)
            indices[f"unsw_repeat{repeat}"] = ui
            indices[f"cicids_repeat{repeat}"] = ci
            u, u_control = torch.from_numpy(source[ui]).double().split(n)
            c, c_control = torch.from_numpy(target[ci]).double().split(n)
            bandwidth = estimate_bandwidth_squared(u, c)
            forward = float(fixed_kernel_mmd(u, c, bandwidth))
            reverse = float(fixed_kernel_mmd(c, u, bandwidth))
            difference = abs(forward - reverse)
            if difference >= epsilon:
                raise AssertionError(f"MMD symmetry failed: {difference} >= {epsilon}")
            variants = {
                'without_byte_features': (u[:, :3], c[:, :3]),
                'without_duration': (u[:, 1:], c[:, 1:]),
            }
            for name, (a, b) in (alternative_spaces or {}).items():
                variants[name] = (torch.from_numpy(a[ui[:n]]).double(),
                                  torch.from_numpy(b[ci[:n]]).double())
            for name, (a, b) in variants.items():
                sigma2 = estimate_bandwidth_squared(a, b)
                value = float(fixed_kernel_mmd(a, b, sigma2))
                sensitivities.setdefault(name, []).append(value)
            for sigma_multiplier in (0.5, 2.0):
                value = float(fixed_kernel_mmd(u, c, bandwidth * sigma_multiplier ** 2))
                sensitivities.setdefault(f'bandwidth_sigma_x{sigma_multiplier}', []).append(value)
            records.append({
                "unsw_to_cicids": forward, "cicids_to_unsw": reverse,
                "absolute_symmetry_error": difference,
                "bandwidth_squared": float(bandwidth),
                "unsw_same_domain": float(fixed_kernel_mmd(u, u_control, bandwidth)),
                "cicids_same_domain": float(fixed_kernel_mmd(c, c_control, bandwidth)),
                "identical_sample_mmd2": float(fixed_kernel_mmd(u, u, bandwidth)),
            })
    values = np.array([r["unsw_to_cicids"] for r in records])
    return {
        "estimator": "biased empirical single-RBF MMD squared",
        "sample_size_per_domain": n, "repeats": repeats,
        "mmd2_mean": float(values.mean()),
        "mmd2_std": float(values.std(ddof=1)) if repeats > 1 else 0.0,
        "std_interpretation": "subsampling variability; not a 95% confidence interval",
        "symmetry_epsilon": epsilon, "symmetry_passed": True,
        "runs": records,
        "sensitivity": {
            name: {'mmd2_runs': runs, 'mmd2_mean': float(np.mean(runs)),
                   'mmd2_std': float(np.std(runs, ddof=1)) if repeats > 1 else 0.0}
            for name, runs in sensitivities.items()
        },
        "sensitivity_protocol": "same fixed subsample indices; feature/transform variants reestimate pooled bandwidth; bandwidth variants use baseline sigma x0.5/x2",
    }, indices


def domain_auc_cv(source, target, seed=42, folds=5):
    """Shared full grouped-CV protocol for every domain-classifier diagnostic."""
    if folds < 2:
        raise ValueError('Grouped domain AUC requires at least two folds')
    x = np.concatenate([source, target]).astype(np.float64)
    y = np.concatenate([np.zeros(len(source), dtype=int), np.ones(len(target), dtype=int)])
    # Treat identical vectors (including missing-value locations) as one group.
    canonical = np.where(np.isfinite(x), x, np.nan)
    canonical[canonical == 0] = 0  # +/-0 are the same vector.
    x = canonical
    keys = np.ascontiguousarray(canonical).view(
        np.dtype((np.void, canonical.dtype.itemsize * canonical.shape[1]))
    ).ravel()
    _, groups = np.unique(keys, return_inverse=True)
    if len(np.unique(groups)) < folds or min(len(source), len(target)) < folds:
        raise ValueError('Not enough samples/groups for the requested domain AUC folds')
    cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores, details, indices = [], [], {}
    for fold, (train, test) in enumerate(cv.split(x, y, groups)):
        if np.intersect1d(groups[train], groups[test]).size:
            raise AssertionError("Duplicate feature vectors overlap train/test")
        if len(np.unique(y[train])) != 2 or len(np.unique(y[test])) != 2:
            raise ValueError("Not enough distinct groups for two-domain grouped CV")
        model = make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            RobustScaler(), LogisticRegression(max_iter=1000, random_state=seed),
        )
        model.fit(x[train], y[train])
        scores.append(float(roc_auc_score(y[test], model.predict_proba(x[test])[:, 1])))
        indices[f'fold{fold}_train'], indices[f'fold{fold}_test'] = train, test
        details.append({'fold': fold, 'train_domain_counts': np.bincount(y[train], minlength=2).tolist(),
                        'test_domain_counts': np.bincount(y[test], minlength=2).tolist()})
    return {
        'classifier': 'logistic_regression', 'cv_folds': folds,
        'source_sample_rows': len(source), 'target_sample_rows': len(target),
        'auc_mean': float(np.mean(scores)), 'auc_std': float(np.std(scores, ddof=1)),
        'auc_folds': scores, 'fold_details': details,
        "split": "full stratified grouped cross-validation",
        "grouping": "identical feature vectors; no overlap between train/test",
        "duplicate_groups_overlap": 0,
        "preprocessing_fit": "classifier training partition only",
        'std_interpretation': 'fold variability; not a 95% confidence interval',
    }, indices


def run_intrinsic(raw_root, seed=42, analysis_sample=100_000,
                  mmd_sample=2048, mmd_repeats=5, overwrite=False):
    """Diagnostic only: canonical raw training samples, shared pooled transform."""
    def signed_log1p(x):
        x = np.asarray(x, dtype=np.float64)
        return np.sign(x) * np.log1p(np.abs(x))
    revision = verify_revision()
    raw_root = Path(raw_root)
    if revision and raw_root.resolve() != revision_path('common_root', raw_root).resolve():
        raise ValueError('Intrinsic raw-root must match the selected frozen canonical common root')
    output_dir = RESULT_ROOT / "intrinsic" / f"seed{seed}"
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"Result exists: {output_dir}. Use --overwrite to rerun.")
    raw, totals, artifacts, provenance = {}, {}, {}, {}
    for offset, domain in enumerate(("unsw", "cicids")):
        path = raw_root / f"{domain}_train"
        print(f"Sampling raw unlabeled training features: {path}", flush=True)
        raw[domain], totals[domain], selected = sample_features(
            path, analysis_sample, seed + offset,
            allow_nonfinite=True, return_indices=True,
        )
        raw[domain] = np.where(np.isfinite(raw[domain]), raw[domain], np.nan)
        artifacts[f"{domain}_raw"] = raw[domain]
        artifacts[f"{domain}_selected_global_rows"] = selected
        provenance[domain] = [
            {"path": str(p.resolve()), "sha256": sha256(p)} for p in parquet_files(path)
        ]
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    pooled = np.concatenate([raw["unsw"], raw["cicids"]])
    logged = signed_log1p(imputer.fit_transform(pooled)).astype(np.float64)
    scaler = RobustScaler().fit(logged)
    transformed = scaler.transform(logged)
    u, c = np.split(transformed, [len(raw["unsw"])])
    print("Computing fixed-pair MMD and symmetry controls...", flush=True)
    log_u, log_c = np.split(logged, [len(raw['unsw'])])
    mmd, pairs = intrinsic_mmd(u, c, mmd_sample, mmd_repeats, seed,
                               {'signed_log_without_scaler': (log_u, log_c)})
    artifacts.update(pairs)
    artifacts.update(unsw_transformed=u, cicids_transformed=c)
    # AUC sees common signed-log features; imputation/scaling fit inside holdout.
    logged_raw = {d: signed_log1p(x).astype(np.float64) for d, x in raw.items()}
    auc, auc_indices = domain_auc_cv(logged_raw["unsw"], logged_raw["cicids"], seed)
    artifacts.update({f"domain_auc_{k}": v for k, v in auc_indices.items()})
    controls = {}
    for offset, domain in enumerate(("unsw", "cicids")):
        permutation = np.random.default_rng(seed + 100 + offset).permutation(len(raw[domain]))
        half = len(permutation) // 2
        first, second = permutation[:half], permutation[half:]
        controls[domain], split_indices = domain_auc_cv(
            logged_raw[domain][first], logged_raw[domain][second], seed + offset,
        )
        artifacts[f"{domain}_auc_random_partition"] = permutation
        artifacts.update({f"{domain}_control_auc_{k}": v for k, v in split_indices.items()})
    result = {
        "protocol": "proposal_v2", "analysis": "intrinsic_domain_shift_diagnostic",
        **revision,
        "seed": seed, "features": list(COMMON_FEATURES),
        "raw_root": str(raw_root.resolve()), "training_rows": totals,
        "sample_rows": {d: len(x) for d, x in raw.items()},
        "sampling": "fixed UNSW seed, CICIDS seed+1; without replacement; reused for both directions",
        "intrusion_labels_used": False, "provenance": provenance,
        "shared_transformation": {
            "fit_role": "pooled unlabeled sampled training data; diagnostic only",
            "steps": "median imputation -> signed_log1p -> RobustScaler",
            "medians": imputer.statistics_.tolist(),
            "centers": scaler.center_.tolist(), "scales": scaler.scale_.tolist(),
        },
        "input_mmd": mmd, "ks_by_feature": compute_ks(u, c),
        "domain_classifier": auc, "same_domain_auc_controls": controls,
        "control_interpretation": "independent same-domain biased MMD need not equal zero; random-partition AUC should be near 0.5, without a hard pass threshold",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = output_dir / "samples.npz"
    np.savez_compressed(samples_path, **artifacts)
    result["sample_artifact"] = str(samples_path.resolve())
    result["sample_artifact_sha256"] = sha256(samples_path)
    output = output_dir / "result.json"
    with output.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"MMD² = {mmd['mmd2_mean']:.6f} ± {mmd['mmd2_std']:.6f} (subsampling std, not 95% CI)")
    print(f"Symmetry passed (epsilon={mmd['symmetry_epsilon']:.3g}); grouped-CV AUC={auc['auc_mean']:.6f} ± {auc['auc_std']:.6f}")
    print(f"Saved: {output}")
    return result


def main():
    parser = argparse.ArgumentParser(
        description=("Pre-adaptation domain shift " "diagnostics for proposal suite.")
    )

    parser.add_argument(
        "--direction",
        default="both",
        choices=[
            "unsw_to_cicids",
            "cicids_to_unsw",
            "both",
        ],
    )
    parser.add_argument("--mode", choices=["directional", "intrinsic"], default="directional")
    parser.add_argument("--raw-root", type=Path, default=revision_path('common_root', ROOT / 'data/bigdata/thesis_20261005/common'),
                        help="Canonical unscaled common-feature training Parquets for intrinsic mode.")
    parser.add_argument("--overwrite", action="store_true", help="Replace intrinsic diagnostic artifacts.")

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--analysis-sample",
        type=int,
        default=100_000,
        help=("Maximum rows sampled per " "domain for KS/domain AUC."),
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
        parser.error("analysis-sample must be >= 2")

    if args.mmd_sample < 2:
        parser.error("mmd-sample must be >= 2")

    if args.mmd_repeats < 1:
        parser.error("mmd-repeats must be >= 1")

    if args.cv_folds < 2:
        parser.error("cv-folds must be >= 2")

    if args.mode == "intrinsic":
        run_intrinsic(args.raw_root, args.seed, args.analysis_sample,
                      args.mmd_sample, args.mmd_repeats, args.overwrite)
        return

    if args.direction == "both":

        for direction in (
            "unsw_to_cicids",
            "cicids_to_unsw",
        ):
            run(
                direction,
                seed=args.seed,
                analysis_sample=args.analysis_sample,
                mmd_sample=args.mmd_sample,
                mmd_repeats=args.mmd_repeats,
                cv_folds=args.cv_folds,
            )

    else:

        run(
            args.direction,
            seed=args.seed,
            analysis_sample=args.analysis_sample,
            mmd_sample=args.mmd_sample,
            mmd_repeats=args.mmd_repeats,
            cv_folds=args.cv_folds,
        )


if __name__ == "__main__":
    main()
