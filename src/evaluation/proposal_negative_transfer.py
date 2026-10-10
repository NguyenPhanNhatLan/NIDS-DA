"""Post-training, development-only negative-transfer diagnostics for proposal_v2."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from torch import nn

from evaluation.baseline import collect_scores, compute_metrics, select_f1_threshold
from evaluation.proposal_domain_shift import compute_domain_auc
from features.common_features import COMMON_FEATURES
from features.parquet_vectors import vector_matrix
from models.baseline import BaselineMLP
from training.adaptation import estimate_bandwidth_squared, mmd_loss, rbf_kernel
from training.proposal_data import ParquetBatchStream, split_sha256
from training.proposal_mmd import freeze_bn_stats, output_paths

from training.data_revision import revision_path, verify_revision

ROOT = Path(__file__).resolve().parents[2]
FEATURE_ROOT = revision_path("feature_root", ROOT / "data/features/proposal_v2")
SOURCE_ROOT = (
    revision_path("model_root", ROOT / "models/proposal_v2") / "source_only_target_val"
)
SOURCE_RESULT_ROOT = (
    revision_path("result_root", ROOT / "results/proposal_v2")
    / "source_only_target_val"
)
OUTPUT_ROOT = (
    revision_path("result_root", ROOT / "results/proposal_v2")
    / "diagnostics_target_val"
)
EPS = 1e-12
RULES = {
    "minimum_ap_drop": 0.01,
    "mmd_reduction": 0.20,
    "separation_ratio_retention": 0.80,
    "gradient_negative_fraction": 0.60,
    "weak_alignment_auc": 0.80,
    "possible_over_alignment_auc": 0.65,
    "prior_gap": 0.10,
    "source_specialization_ap_gain": 0.05,
    "bn_shift_ratio": 2.0,
    "bn_shift_absolute_gap": 0.5,
    "threshold_f1_gap": 0.10,
    "threshold_min_roc_auc": 0.70,
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def domains(direction):
    if direction == "unsw_to_cicids":
        return "unsw", "cicids"
    if direction == "cicids_to_unsw":
        return "cicids", "unsw"
    raise ValueError(f"Unknown direction: {direction}")


def load_models(direction, seed, config_path):
    source_path = SOURCE_ROOT / direction / f"seed{seed}.pt"
    adapted_path, result_path = output_paths(direction, seed, config_path)
    source_result_path = SOURCE_RESULT_ROOT / direction / f"seed{seed}.json"
    for path in (source_path, adapted_path, source_result_path, result_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required experiment artifact missing: {path}")
    source_cp = torch.load(source_path, map_location="cpu", weights_only=True)
    adapted_cp = torch.load(adapted_path, map_location="cpu", weights_only=True)
    source_result = json.loads(source_result_path.read_text())
    adapted_result = json.loads(result_path.read_text())
    for artifact in (source_cp, adapted_cp, source_result, adapted_result):
        if (
            artifact["direction"] != direction
            or artifact["seed"] != seed
            or artifact["features"] != list(COMMON_FEATURES)
        ):
            raise ValueError("Experiment direction, seed, or feature schema mismatch")
    if adapted_cp["source_checkpoint_sha256"] != sha256(source_path):
        raise ValueError(
            "Adapted checkpoint does not derive from current source checkpoint"
        )
    common_hash = sha256(ROOT / "configs/common_features_v2.json")
    preprocessor_hash = sha256(
        revision_path("model_root", ROOT / "models/proposal_v2")
        / direction
        / "preprocessor.joblib"
    )
    for artifact in (source_cp, adapted_cp, source_result, adapted_result):
        if (
            artifact["common_feature_config_sha256"] != common_hash
            or artifact["preprocessor_sha256"] != preprocessor_hash
        ):
            raise ValueError("Experiment artifact preprocessing provenance mismatch")
    source, target = domains(direction)
    base = FEATURE_ROOT / direction
    current_splits = {
        "source_train": split_sha256(base / f"{source}_train"),
        "source_val": split_sha256(base / f"{source}_val"),
        "target_train": split_sha256(base / f"{target}_train"),
        "target_val": split_sha256(base / f"{target}_val"),
    }
    for artifact in (source_cp, adapted_cp, source_result, adapted_result):
        if artifact["prepared_split_sha256"] != current_splits:
            raise ValueError("Experiment artifact prepared split provenance mismatch")
    if (
        source_result["target_development_split"] != f"{target}_val"
        or adapted_result["target_development_split"] != f"{target}_val"
    ):
        raise ValueError("Experiments must use target validation as development data")
    if adapted_result["config_sha256"] != sha256(config_path):
        raise ValueError("Adapted result config hash differs from selected config")
    if adapted_result["checkpoint"] != str(adapted_path):
        raise ValueError("Adapted result refers to a different checkpoint")
    models = []
    for cp in (source_cp, adapted_cp):
        model = BaselineMLP(len(COMMON_FEATURES))
        model.load_state_dict(cp["model_state_dict"])
        model.eval()
        models.append(model)
    paths = {
        "source_checkpoint": str(source_path),
        "adapted_checkpoint": str(adapted_path),
        "source_result": str(source_result_path),
        "adapted_result": str(result_path),
        "source_checkpoint_sha256": sha256(source_path),
        "adapted_checkpoint_sha256": sha256(adapted_path),
    }
    return models[0], models[1], source_result, adapted_result, paths


def sample_labeled(directory, max_rows, seed):
    files = sorted(Path(directory).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet files in {directory}")
    total = sum(pq.read_metadata(path).num_rows for path in files)
    if total < 2:
        raise ValueError(f"Too few rows in {directory}")
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(total, size=min(max_rows, total), replace=False))
    xs, ys = [], []
    offset = 0
    for path in files:
        with pq.ParquetFile(path) as parquet:
            for batch in parquet.iter_batches(
                batch_size=65536, columns=["features", "label"]
            ):
                n = len(batch)
                left = np.searchsorted(selected, offset)
                right = np.searchsorted(selected, offset + n)
                if right > left:
                    indices = selected[left:right] - offset
                    vector = batch.column("features")
                    x = vector_matrix(vector, path)
                    y = np.asarray(
                        batch.column("label").to_numpy(zero_copy_only=False),
                        dtype=np.int64,
                    )
                    xs.append(x[indices].copy())
                    ys.append(y[indices].copy())
                offset += n
    x, y = np.concatenate(xs), np.concatenate(ys)
    if (
        len(x) != len(selected)
        or not np.isfinite(x).all()
        or not np.isin(y, [0, 1]).all()
    ):
        raise ValueError(f"Invalid sampled development rows: {directory}")
    return x, y


def collect_representation(model, x, batch_size=1024):
    sums = {"bn1": None, "bn2": None}
    counts = {"bn1": 0, "bn2": 0}
    hooks = []
    for name in sums:

        def capture(module, args, name=name):
            values = args[0].detach().double()
            part = values.sum(dim=0).cpu()
            sums[name] = part if sums[name] is None else sums[name] + part
            counts[name] += len(values)

        hooks.append(getattr(model, name).register_forward_pre_hook(capture))
    parts = []
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(x), batch_size):
                z, _ = model(torch.from_numpy(x[start : start + batch_size]))
                parts.append(z.cpu().numpy())
    finally:
        for hook in hooks:
            hook.remove()
    means = {name: (sums[name] / counts[name]).numpy() for name in sums}
    return np.concatenate(parts), means


def evaluate_model(
    model, source_val_path, target_development_path, source_threshold, input_dim
):
    source_labels, source_scores = collect_scores(
        model, ParquetBatchStream(source_val_path, 1024, False, 0, True)
    )
    target_labels, target_scores = collect_scores(
        model, ParquetBatchStream(target_development_path, 1024, False, 0, True)
    )
    return {
        "source_val": compute_metrics(source_labels, source_scores, source_threshold),
        "target_development": compute_metrics(
            target_labels, target_scores, source_threshold
        ),
        "source_labels": source_labels,
        "target_labels": target_labels,
        "target_scores": target_scores,
    }


def repeated_mmd(source_z, target_z, sample_size=256, repeats=20, seed=42):
    n = min(sample_size, len(source_z), len(target_z))
    if n < 2 or repeats < 1:
        raise ValueError("MMD needs at least two samples and one repeat")
    rng = np.random.default_rng(seed)
    values, bandwidths = [], []
    with torch.no_grad():
        for _ in range(repeats):
            a = torch.from_numpy(source_z[rng.choice(len(source_z), n, replace=False)])
            b = torch.from_numpy(target_z[rng.choice(len(target_z), n, replace=False)])
            loss, bandwidth = mmd_loss(a, b)
            values.append(float(loss))
            bandwidths.append(float(bandwidth))
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if repeats > 1 else 0.0
    return {
        "sample_size_per_domain": n,
        "repeats": repeats,
        "mmd2_mean": mean,
        "mmd2_std": std,
        "cv": std / max(abs(mean), EPS),
        "bandwidth_mean": float(np.mean(bandwidths)),
        "bandwidth_std": float(np.std(bandwidths, ddof=1)) if repeats > 1 else 0.0,
    }


def class_conditional_mmd(
    source_z, source_y, target_z, target_y, sample_size=128, seed=42
):
    groups = [
        source_z[source_y == 0],
        source_z[source_y == 1],
        target_z[target_y == 0],
        target_z[target_y == 1],
    ]
    n = min(sample_size, *(len(group) for group in groups))
    if n < 2:
        raise ValueError(
            "Class-conditional MMD needs both classes in both sampled domains"
        )
    rng = np.random.default_rng(seed)
    sn, sa, tn, ta = [
        torch.from_numpy(group[rng.choice(len(group), n, replace=False)])
        for group in groups
    ]
    with torch.no_grad():
        bandwidth_squared = estimate_bandwidth_squared(
            torch.cat([sn, sa]), torch.cat([tn, ta])
        )

        def shared_mmd(a, b):
            return float(
                (
                    rbf_kernel(a, a, bandwidth_squared).mean()
                    + rbf_kernel(b, b, bandwidth_squared).mean()
                    - 2 * rbf_kernel(a, b, bandwidth_squared).mean()
                ).item()
            )

        nn = shared_mmd(sn, tn)
        aa = shared_mmd(sa, ta)
        na = shared_mmd(sn, ta)
        an = shared_mmd(sa, tn)
    same, cross = (nn + aa) / 2, (na + an) / 2
    return {
        "sample_size_per_class": n,
        "shared_bandwidth": float(torch.sqrt(bandwidth_squared)),
        "normal_normal": nn,
        "attack_attack": aa,
        "normal_attack": na,
        "attack_normal": an,
        "same_class": same,
        "cross_class": cross,
        "separation_ratio": cross / max(same, EPS),
    }


def gradient_cosine(
    model,
    source_train_path,
    target_train_path,
    input_dim,
    counts,
    batches=20,
    batch_size=256,
):
    source_loader = ParquetBatchStream(source_train_path, batch_size, False, 0, True)
    target_loader = ParquetBatchStream(
        target_train_path, batch_size, True, 1000, False, drop_last=True
    )
    params = [
        *model.fc1.parameters(),
        *model.bn1.parameters(),
        *model.fc2.parameters(),
        *model.bn2.parameters(),
    ]
    class_counts = torch.as_tensor(counts, dtype=torch.float32)
    weights = class_counts.sum() / (2 * class_counts)
    criterion = nn.CrossEntropyLoss(weight=weights)
    training_flags = {module: module.training for module in model.modules()}
    model.train()
    freeze_bn_stats(model)
    cosines = []
    try:
        target_iter = iter(target_loader)
        for source_x, source_y in source_loader:
            if len(cosines) >= batches:
                break
            target_x = next(target_iter)
            n = min(len(source_x), len(target_x))
            if n < 2:
                continue
            z, logits = model(torch.cat([source_x[:n], target_x[:n]]))
            ce = criterion(logits[:n], source_y[:n])
            mmd = mmd_loss(z[:n], z[n:])[0]
            g_ce = torch.autograd.grad(ce, params, retain_graph=True, allow_unused=True)
            g_mmd = torch.autograd.grad(mmd, params, allow_unused=True)
            flat_ce = torch.cat(
                [
                    torch.zeros_like(p).flatten() if g is None else g.flatten()
                    for p, g in zip(params, g_ce)
                ]
            )
            flat_mmd = torch.cat(
                [
                    torch.zeros_like(p).flatten() if g is None else g.flatten()
                    for p, g in zip(params, g_mmd)
                ]
            )
            if flat_ce.norm() > EPS and flat_mmd.norm() > EPS:
                cosines.append(float(F.cosine_similarity(flat_ce, flat_mmd, dim=0)))
    finally:
        for module, was_training in training_flags.items():
            module.training = was_training
    if not cosines:
        raise ValueError("No usable batches for gradient cosine")
    return {
        "batches": len(cosines),
        "mean": float(np.mean(cosines)),
        "std": float(np.std(cosines, ddof=1)) if len(cosines) > 1 else 0.0,
        "negative_fraction": float(np.mean(np.asarray(cosines) < 0)),
    }


def latent_domain_auc(source_z, target_z, seed=42, folds=5):
    return compute_domain_auc(source_z, target_z, seed=seed, folds=folds)


def label_prior_shift(source_y, target_y):
    source_prior = float(np.mean(source_y))
    target_prior = float(np.mean(target_y))
    return {
        "source_attack_prevalence": source_prior,
        "target_attack_prevalence": target_prior,
        "absolute_gap": abs(source_prior - target_prior),
    }


def bn_shift(model, source_means, target_means):
    result = {}
    for name in ("bn1", "bn2"):
        bn = getattr(model, name)
        scale = np.sqrt(bn.running_var.detach().cpu().numpy() + bn.eps)
        center = bn.running_mean.detach().cpu().numpy()
        source_shift = float(np.mean(np.abs(source_means[name] - center) / scale))
        target_shift = float(np.mean(np.abs(target_means[name] - center) / scale))
        result[name] = {
            "source_shift": source_shift,
            "target_shift": target_shift,
            "target_minus_source": target_shift - source_shift,
        }
    return result


def threshold_diagnostic(
    labels, scores, source_threshold, applied_metrics, min_roc_auc=0.70, min_f1_gap=0.10
):
    oracle = select_f1_threshold(labels, scores)
    oracle_metrics = compute_metrics(labels, scores, oracle)
    gap = oracle_metrics["f1"] - applied_metrics["f1"]
    return {
        "target_ap": applied_metrics["pr_auc"],
        "target_roc_auc": applied_metrics["roc_auc"],
        "source_threshold": source_threshold,
        "f1_source_threshold": applied_metrics["f1"],
        "target_oracle_threshold_development_only": oracle,
        "target_oracle_f1_development_only": oracle_metrics["f1"],
        "oracle_minus_source_f1": gap,
        "threshold_issue": bool(
            applied_metrics["roc_auc"] >= min_roc_auc
            and applied_metrics["pr_auc"]
            >= max(0.30, 2 * applied_metrics["prevalence"])
            and gap >= min_f1_gap
        ),
    }


def diagnose(
    target_delta,
    source_delta,
    marginal,
    class_alignment,
    gradient,
    domain,
    prior,
    threshold,
    mmd_active=True,
    bn=None,
):
    ap_drop = target_delta["pr_auc"]
    reduction = marginal["relative_mmd_reduction"]
    before_ratio = class_alignment["before"]["separation_ratio"]
    after_ratio = class_alignment["after"]["separation_ratio"]
    causes = []
    if (
        mmd_active
        and reduction > RULES["mmd_reduction"]
        and domain["after"]["auc_mean"] > RULES["weak_alignment_auc"]
    ):
        causes.append("mmd_reduced_but_domains_still_separable")
    if after_ratio < before_ratio * RULES["separation_ratio_retention"]:
        causes.append("class_mixing")
    if (
        mmd_active
        and gradient["mean"] < 0
        and gradient["negative_fraction"] > RULES["gradient_negative_fraction"]
    ):
        causes.append("ce_mmd_gradient_conflict")
    if mmd_active and domain["after"]["auc_mean"] > RULES["weak_alignment_auc"]:
        causes.append("weak_alignment")
    if (
        mmd_active
        and domain["after"]["auc_mean"] < RULES["possible_over_alignment_auc"]
        and ap_drop < -RULES["minimum_ap_drop"]
    ):
        causes.append("possible_over_alignment")
    if (
        source_delta > RULES["source_specialization_ap_gain"]
        and ap_drop < -RULES["minimum_ap_drop"]
    ):
        causes.append("source_over_specialization")
    if bn is not None and any(
        values["target_shift"] > values["source_shift"] * RULES["bn_shift_ratio"]
        and values["target_minus_source"] > RULES["bn_shift_absolute_gap"]
        for values in bn["after"].values()
    ):
        causes.append("normalization_shift")
    if prior["absolute_gap"] > RULES["prior_gap"]:
        causes.append("label_prior_shift")
    if threshold["after"]["threshold_issue"]:
        causes.append("threshold_mismatch")
    return causes


def run(
    direction,
    seed=42,
    config_path=ROOT / "configs/proposal_mmd_v2.json",
    sample_rows=10000,
    mmd_sample=256,
    mmd_repeats=20,
    gradient_batches=20,
    cv_folds=5,
):
    revision = verify_revision()
    config_path = Path(config_path)
    source, target = domains(direction)
    default_config = ROOT / "configs/proposal_mmd_v2.json"
    filename = (
        f"seed{seed}.json"
        if config_path.resolve() == default_config.resolve()
        else f"seed{seed}_{config_path.stem}.json"
    )
    output = OUTPUT_ROOT / direction / filename
    source_model, adapted_model, source_result, adapted_result, paths = load_models(
        direction, seed, config_path
    )
    base = FEATURE_ROOT / direction
    source_val_path = base / f"{source}_val"
    target_development_path = base / f"{target}_val"
    source_train_path = base / f"{source}_train"
    target_train_path = base / f"{target}_train"
    input_dim = len(COMMON_FEATURES)
    print(f"Sampling development rows for {direction}...", flush=True)
    source_x, source_y = sample_labeled(source_val_path, sample_rows, seed)
    target_x, target_y = sample_labeled(target_development_path, sample_rows, seed + 1)
    if any(np.sum(y == cls) < 2 for y in (source_y, target_y) for cls in (0, 1)):
        raise ValueError(
            "Increase --sample-rows so both classes are present in both development samples"
        )

    print(
        "Evaluating both checkpoints on full source validation and target development...",
        flush=True,
    )
    baseline_eval = evaluate_model(
        source_model,
        source_val_path,
        target_development_path,
        source_result["threshold_from_source_val"],
        input_dim,
    )
    adapted_eval = evaluate_model(
        adapted_model,
        source_val_path,
        target_development_path,
        adapted_result["threshold"],
        input_dim,
    )
    from evaluation.proposal_score_diagnostic import score_diagnostics

    score_analysis = {}

    for name, evaluated, threshold in (
        (
            "before",
            baseline_eval,
            source_result["threshold_from_source_val"],
        ),
        (
            "after",
            adapted_eval,
            adapted_result["threshold"],
        ),
    ):
        score_analysis[name] = score_diagnostics(
            evaluated["target_labels"],
            evaluated["target_scores"],
            threshold,
        )

    if (
        abs(
            baseline_eval["target_development"]["pr_auc"]
            - source_result["target_development"]["pr_auc"]
        )
        > 1e-3
    ):
        raise ValueError("Source-only target AP differs from saved experiment result")
    if (
        abs(
            adapted_eval["target_development"]["pr_auc"]
            - adapted_result["cross_domain"]["pr_auc"]
        )
        > 1e-3
    ):
        raise ValueError("Adapted target AP differs from saved experiment result")
    target_delta = {
        key: adapted_result["cross_domain"][key]
        - source_result["target_development"][key]
        for key in ("pr_auc", "roc_auc", "macro_f1", "recall", "fpr")
    }
    source_delta = (
        adapted_eval["source_val"]["pr_auc"] - baseline_eval["source_val"]["pr_auc"]
    )

    latent = {}
    bn_means = {}
    print("Collecting latent representations and BN activations...", flush=True)
    for name, model in (("before", source_model), ("after", adapted_model)):
        sz, source_means = collect_representation(model, source_x)
        tz, target_means = collect_representation(model, target_x)
        latent[name] = (sz, tz)
        bn_means[name] = (source_means, target_means)
    mmd = {
        name: repeated_mmd(*latent[name], mmd_sample, mmd_repeats, seed)
        for name in ("before", "after")
    }
    reduction = (mmd["before"]["mmd2_mean"] - mmd["after"]["mmd2_mean"]) / max(
        mmd["before"]["mmd2_mean"], EPS
    )
    marginal = {
        "before": mmd["before"],
        "after": mmd["after"],
        "relative_mmd_reduction": reduction,
    }
    class_alignment = {
        name: class_conditional_mmd(
            latent[name][0],
            source_y,
            latent[name][1],
            target_y,
            sample_size=min(mmd_sample, 128),
            seed=seed,
        )
        for name in ("before", "after")
    }
    domain = {
        name: latent_domain_auc(*latent[name], seed=seed, folds=cv_folds)
        for name in ("before", "after")
    }
    print("Measuring gradient cosine on source/target training features...", flush=True)
    torch.manual_seed(seed)
    gradient = gradient_cosine(
        adapted_model,
        source_train_path,
        target_train_path,
        input_dim,
        source_result["source_train_counts"],
        batches=gradient_batches,
    )
    prior = label_prior_shift(
        baseline_eval["source_labels"], baseline_eval["target_labels"]
    )
    bn = {
        name: bn_shift(model, *bn_means[name])
        for name, model in (("before", source_model), ("after", adapted_model))
    }
    threshold = {
        "before": threshold_diagnostic(
            baseline_eval["target_labels"],
            baseline_eval["target_scores"],
            source_result["threshold_from_source_val"],
            baseline_eval["target_development"],
        ),
        "after": threshold_diagnostic(
            adapted_eval["target_labels"],
            adapted_eval["target_scores"],
            adapted_result["threshold"],
            adapted_eval["target_development"],
        ),
    }
    mmd_active = float(adapted_result["lambda_mmd"]) > 0
    causes = diagnose(
        target_delta,
        source_delta,
        marginal,
        class_alignment,
        gradient,
        domain,
        prior,
        threshold,
        mmd_active,
        bn,
    )
    ce_control = None
    result = {
        "direction": direction,
        **revision,
        "seed": seed,
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "artifacts": paths,
        "sampling": {
            "development_rows_requested_per_domain": sample_rows,
            "source_development_rows_sampled": len(source_x),
            "target_development_rows_sampled": len(target_x),
            "source_sample_class_counts": np.bincount(source_y, minlength=2).tolist(),
            "target_sample_class_counts": np.bincount(target_y, minlength=2).tolist(),
            "mmd_sample": mmd_sample,
            "mmd_repeats": mmd_repeats,
            "gradient_batches": gradient_batches,
            "cv_folds": cv_folds,
        },
        "negative_transfer": {
            "detected": bool(target_delta["pr_auc"] < 0),
            "delta_target_pr_auc": target_delta["pr_auc"],
            "target_delta": target_delta,
            "source_only_target": {
                k: source_result["target_development"][k] for k in target_delta
            },
            "adapted_target": {
                k: adapted_result["cross_domain"][k] for k in target_delta
            },
        },
        "source_preservation": {
            "source_val_ap_before": baseline_eval["source_val"]["pr_auc"],
            "source_val_ap_after": adapted_eval["source_val"]["pr_auc"],
            "source_val_ap_delta": source_delta,
            "preserved_or_improved": bool(source_delta >= 0),
        },
        "marginal_alignment": marginal,
        "class_alignment": {
            "before": class_alignment["before"],
            "after": class_alignment["after"],
            "separation_ratio_delta": class_alignment["after"]["separation_ratio"]
            - class_alignment["before"]["separation_ratio"],
            "class_mixing_heuristic": bool(
                class_alignment["after"]["separation_ratio"]
                < class_alignment["before"]["separation_ratio"]
                * RULES["separation_ratio_retention"]
            ),
        },
        "gradient_conflict": {
            **gradient,
            "mmd_active_in_training": mmd_active,
            "conflict_heuristic": bool(
                mmd_active
                and gradient["mean"] < 0
                and gradient["negative_fraction"] > RULES["gradient_negative_fraction"]
            ),
        },
        "target_validation_score_analysis": score_analysis,
        "domain_separability": domain,
        "label_prior": prior,
        "batchnorm_shift": bn,
        "threshold": threshold,
        "mmd_stability": {"before": mmd["before"], "after": mmd["after"]},
        "continued_ce_control": ce_control,
        "likely_causes": causes,
        "diagnostic_rules": RULES,
        "diagnostic_rules_are_heuristic": True,
        "diagnostic_only": True,
        "post_hoc": True,
        "target_labels_used_training": False,
        "target_labels_used_checkpoint_selection": False,
        "target_labels_used_threshold_selection": False,
        "data_roles": {
            "source_labeled_diagnostics": f"{source}_val",
            "target_labeled_diagnostics": f"{target}_val (development)",
            "source_gradient": f"{source}_train",
            "target_gradient_features_only": f"{target}_train",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print_summary(result, output)
    return result


def print_summary(result, output):
    transfer = result["negative_transfer"]
    print("\n=== Negative Transfer Diagnostic ===")
    print(
        f"Target AP: source-only={transfer['source_only_target']['pr_auc']:.4f}, "
        f"adapted={transfer['adapted_target']['pr_auc']:.4f}, "
        f"delta={transfer['delta_target_pr_auc']:+.4f}"
    )
    print(f"NEGATIVE TRANSFER: {'YES' if transfer['detected'] else 'NO'}")
    source = result["source_preservation"]
    print(
        f"[1] Source preservation | Val AP delta={source['source_val_ap_delta']:+.4f}"
    )
    marginal = result["marginal_alignment"]
    print(
        f"[2] Marginal alignment | MMD² {marginal['before']['mmd2_mean']:.4f} -> {marginal['after']['mmd2_mean']:.4f}"
    )
    classes = result["class_alignment"]
    print(
        f"[3] Class structure | ratio {classes['before']['separation_ratio']:.4f} -> {classes['after']['separation_ratio']:.4f}"
    )
    grad = result["gradient_conflict"]
    print(
        f"[4] Gradient conflict | cosine={grad['mean']:+.4f}, negative fraction={grad['negative_fraction']:.2f}"
    )
    domain = result["domain_separability"]
    print(
        f"[5] Domain separability | AUC {domain['before']['auc_mean']:.4f} -> {domain['after']['auc_mean']:.4f}"
    )
    prior = result["label_prior"]
    print(
        f"[6] Label prior | source={prior['source_attack_prevalence']:.4f}, target={prior['target_attack_prevalence']:.4f}"
    )
    bn = result["batchnorm_shift"]["after"]
    print(
        f"[7] BN shift | bn1 source/target={bn['bn1']['source_shift']:.3f}/{bn['bn1']['target_shift']:.3f}; "
        f"bn2={bn['bn2']['source_shift']:.3f}/{bn['bn2']['target_shift']:.3f}"
    )
    threshold = result["threshold"]["after"]
    print(
        f"[8] Threshold | AP={threshold['target_ap']:.4f}, ROC={threshold['target_roc_auc']:.4f}, "
        f"F1 source/oracle={threshold['f1_source_threshold']:.4f}/{threshold['target_oracle_f1_development_only']:.4f}"
    )
    print(f"Likely causes (heuristic): {', '.join(result['likely_causes']) or 'none'}")
    if result["continued_ce_control"] is not None:
        control = result["continued_ce_control"]
        print(
            f"CE-only control | Target AP={control['target_development_ap']:.4f}, "
            f"MMD minus CE-only={control['mmd_minus_ce_only_ap']:+.4f}"
        )
    analysis = result.get("target_validation_score_analysis")
    if analysis:
        print("\n=== Target validation score analysis ===")
        for phase, scores in analysis.items():
            metrics = scores["source_threshold_metrics"]
            oracle = scores["oracle_f1_diagnostic_only"]
            print(
                f"{phase}: AP={scores['average_precision']:.4f}, "
                f"ROC-AUC={scores['roc_auc']:.4f}, "
                f"source threshold={metrics['threshold']:.4f}, "
                f"recall={metrics['recall']:.4f}, FPR={metrics['fpr']:.4f}, "
                f"F1={metrics['f1']:.4f}, validation oracle F1={oracle['f1']:.4f}"
            )
            for label, distribution in scores["score_distributions"].items():
                print(
                    f"  {label}: n={distribution['count']}, "
                    f"median score={distribution['quantiles']['p50']:.4f}"
                )
        print("Full score distributions and PR curves are saved in the diagnostic JSON.")
    print(f"Saved: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--direction", required=True, choices=["unsw_to_cicids", "cicids_to_unsw"]
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/proposal_mmd_v2.json"
    )
    parser.add_argument("--sample-rows", type=int, default=10000)
    parser.add_argument("--mmd-sample", type=int, default=256)
    parser.add_argument("--mmd-repeats", type=int, default=20)
    parser.add_argument("--gradient-batches", type=int, default=20)
    parser.add_argument("--cv-folds", type=int, default=5)
    args = parser.parse_args()
    if (
        args.sample_rows < 10
        or args.mmd_sample < 2
        or args.mmd_repeats < 1
        or args.gradient_batches < 1
        or args.cv_folds < 2
    ):
        parser.error("Invalid diagnostic sample/repeat/fold setting")
    run(
        args.direction,
        args.seed,
        args.config,
        args.sample_rows,
        args.mmd_sample,
        args.mmd_repeats,
        args.gradient_batches,
        args.cv_folds,
    )


if __name__ == "__main__":
    main()
