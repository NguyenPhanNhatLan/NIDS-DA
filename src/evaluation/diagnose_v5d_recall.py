"""Development-only recall diagnostic for V5b vs V5d.

This script DOES NOT train, calibrate, or modify checkpoints.
It reads the already-frozen V5d calibration/checkpoint and uses DEVELOPMENT
labels only for diagnosis.

Important:
- Recall@target-FPR thresholds are DEV ORACLE DIAGNOSTICS ONLY.
- They must NOT be reused as deployment/final-test thresholds.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from evaluation.baseline import compute_metrics
from evaluation.calibration import calibrate_margin, select_threshold_by_fpr
from evaluation.hda_v5b_calibration import collect_margins, data_snapshot, payload_hash
from evaluation.hda_v5d import (
    calibration_path,
    dependencies,
    load_v5b_calibration,
    validate_affine,
    verify_training_data,
)
from training.hda_v5d import load_context, load_student
from training.thesis_protocol import evaluation_target, resolve_path
from training.v6_data import make_loader


FPR_BUDGETS = (0.01, 0.02, 0.03, 0.05)


def _finite_1d(name, values):
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or len(x) == 0 or not np.isfinite(x).all():
        raise ValueError(f"{name} must be a nonempty finite 1D vector")
    return x


def _quantiles(values, qs):
    x = _finite_1d("quantile values", values)
    return {f"q{int(round(q * 100)):02d}": float(np.quantile(x, q)) for q in qs}


def _score_summary(labels, scores, threshold):
    labels = np.asarray(labels, dtype=np.int64)
    scores = _finite_1d("scores", scores)
    if labels.shape != scores.shape:
        raise ValueError("labels/scores shape mismatch")

    normal = scores[labels == 0]
    attack = scores[labels == 1]
    if len(normal) == 0 or len(attack) == 0:
        raise ValueError("Development set must contain both classes")

    attack_below = attack < threshold
    normal_above = normal >= threshold

    return {
        "attack_quantiles": _quantiles(attack, (0.10, 0.25, 0.50, 0.75, 0.90)),
        "normal_quantiles": _quantiles(normal, (0.90, 0.95, 0.97, 0.99)),
        "attack_count": int(len(attack)),
        "normal_count": int(len(normal)),
        "attack_below_current_threshold_count": int(attack_below.sum()),
        "attack_below_current_threshold_fraction": float(attack_below.mean()),
        "normal_above_current_threshold_count": int(normal_above.sum()),
        "normal_above_current_threshold_fraction": float(normal_above.mean()),
    }


def _recall_at_target_fpr(labels, scores):
    """DEV ORACLE ONLY: choose threshold on development labels for each FPR budget."""
    labels = np.asarray(labels, dtype=np.int64)
    scores = _finite_1d("scores", scores)
    rows = {}
    for budget in FPR_BUDGETS:
        threshold = select_threshold_by_fpr(labels, scores, budget)
        metrics = compute_metrics(labels, scores, threshold)
        rows[f"fpr_{int(round(budget * 100)):02d}pct"] = {
            "budget": float(budget),
            "threshold": float(threshold),
            "recall": metrics["recall"],
            "fpr": metrics["fpr"],
            "precision": metrics["precision"],
            "f1": metrics["f1"],
            "tp": metrics["tp"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "tn": metrics["tn"],
        }
    return rows


def _attack_transition(labels, v5b_scores, v5b_threshold, v5d_scores, v5d_threshold):
    labels = np.asarray(labels, dtype=np.int64)
    attack = labels == 1
    n_attack = int(attack.sum())
    if n_attack == 0:
        raise ValueError("No Attack rows in development set")

    b_detect = v5b_scores >= v5b_threshold
    d_detect = v5d_scores >= v5d_threshold

    both_detect = attack & b_detect & d_detect
    v5d_recovered = attack & (~b_detect) & d_detect
    v5d_lost = attack & b_detect & (~d_detect)
    both_missed = attack & (~b_detect) & (~d_detect)

    def row(mask):
        count = int(mask.sum())
        return {"count": count, "fraction_of_attacks": count / n_attack}

    return {
        "attack_total": n_attack,
        "both_detected": row(both_detect),
        "v5d_recovered_vs_v5b": row(v5d_recovered),
        "v5d_lost_vs_v5b": row(v5d_lost),
        "both_missed": row(both_missed),
        "net_attack_gain_v5d_minus_v5b": int(v5d_recovered.sum() - v5d_lost.sum()),
    }


def _normal_transition(labels, v5b_scores, v5b_threshold, v5d_scores, v5d_threshold):
    labels = np.asarray(labels, dtype=np.int64)
    normal = labels == 0
    n_normal = int(normal.sum())
    if n_normal == 0:
        raise ValueError("No Normal rows in development set")

    b_fp = v5b_scores >= v5b_threshold
    d_fp = v5d_scores >= v5d_threshold

    both_correct = normal & (~b_fp) & (~d_fp)
    v5d_fixed_fp = normal & b_fp & (~d_fp)
    v5d_new_fp = normal & (~b_fp) & d_fp
    both_fp = normal & b_fp & d_fp

    def row(mask):
        count = int(mask.sum())
        return {"count": count, "fraction_of_normals": count / n_normal}

    return {
        "normal_total": n_normal,
        "both_correct": row(both_correct),
        "v5d_fixed_v5b_false_positive": row(v5d_fixed_fp),
        "v5d_new_false_positive": row(v5d_new_fp),
        "both_false_positive": row(both_fp),
        "net_fp_change_v5d_minus_v5b": int(v5d_new_fp.sum() - v5d_fixed_fp.sum()),
    }


def _score_shift(labels, v5b_scores, v5d_scores):
    labels = np.asarray(labels, dtype=np.int64)
    delta = _finite_1d("score delta", v5d_scores - v5b_scores)

    def summarize(mask):
        x = delta[mask]
        return {
            "mean": float(np.mean(x)),
            "median": float(np.median(x)),
            "q10": float(np.quantile(x, 0.10)),
            "q25": float(np.quantile(x, 0.25)),
            "q75": float(np.quantile(x, 0.75)),
            "q90": float(np.quantile(x, 0.90)),
            "fraction_score_increased": float(np.mean(x > 0)),
            "fraction_score_decreased": float(np.mean(x < 0)),
        }

    return {
        "normal": summarize(labels == 0),
        "attack": summarize(labels == 1),
    }


def _ranking_agreement(v5b_scores, v5d_scores):
    b = _finite_1d("v5b scores", v5b_scores)
    d = _finite_1d("v5d scores", v5d_scores)
    if b.shape != d.shape:
        raise ValueError("V5b/V5d score shape mismatch")
    if len(b) < 2 or np.ptp(b) == 0 or np.ptp(d) == 0:
        return {"pearson": None, "spearman": None}
    pearson = float(np.corrcoef(b, d)[0, 1])
    spearman = float(spearmanr(b, d).statistic)
    return {"pearson": pearson, "spearman": spearman}


def _print_recall_table(result):
    print("\n=== DEV ORACLE DIAGNOSTIC: Recall @ target FPR ===")
    print("DO NOT use these thresholds for deployment/final test.")
    print("Budget | V5B Recall | V5D Recall | Delta | V5B actual FPR | V5D actual FPR")
    for budget in FPR_BUDGETS:
        key = f"fpr_{int(round(budget * 100)):02d}pct"
        b = result["models"]["v5b"]["recall_at_target_fpr"][key]
        d = result["models"]["v5d"]["recall_at_target_fpr"][key]
        print(
            f"{budget:>5.0%} | "
            f"{b['recall']:.6f} | {d['recall']:.6f} | "
            f"{d['recall'] - b['recall']:+.6f} | "
            f"{b['fpr']:.6f} | {d['fpr']:.6f}"
        )


def _print_current(result):
    print("\n=== Current frozen operating points ===")
    print("Model | AP | ROC-AUC | F1 | Recall | FPR | Threshold")
    for name in ("v5b", "v5d"):
        m = result["models"][name]["current_operating_metrics"]
        print(
            f"{name.upper()} | "
            f"{m['pr_auc']:.6f} | {m['roc_auc']:.6f} | "
            f"{m['f1']:.6f} | {m['recall']:.6f} | {m['fpr']:.6f} | "
            f"{m['threshold']:.6f}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hda_v5d.json")
    parser.add_argument("--training-seed", type=int, choices=(42, 43, 44), default=42)
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output. Default: <result_dir>/diagnose_v5d_recall_seedN.json",
    )
    args = parser.parse_args()

    config, protocol, provenance, source, _, teacher = load_context(
        args.config, args.training_seed
    )
    output = (resolve_path(args.output) if args.output else
              resolve_path(config["result_dir"]) / f"diagnose_v5d_recall_seed{config['training_seed']}.json")
    if output.exists():
        raise FileExistsError(f"Diagnostic already exists: {output}")
    student, checkpoint = load_student(config, provenance, source, teacher)
    verify_training_data(protocol, checkpoint)

    # Validate the exact frozen V5b calibration dependency used by V5d.
    v5b_artifact = load_v5b_calibration(config, protocol, provenance)

    # Validate V5d calibration artifact exactly as the normal evaluator does.
    artifact_path = calibration_path(config)
    artifact = json.loads(artifact_path.read_text())
    frozen = artifact["frozen"]

    if payload_hash(frozen) != artifact["sha256"]:
        raise ValueError("V5d calibration payload changed")
    if frozen["dependencies"] != dependencies(config, provenance):
        raise ValueError("V5d calibration dependencies changed")
    if frozen["v5b_calibration_payload_sha256"] != v5b_artifact["sha256"]:
        raise ValueError("V5b calibration dependency changed")
    validate_affine(frozen["parameters"]["v5b"])
    validate_affine(frozen["parameters"]["v5d"])
    if (frozen["source_validation_files"] != data_snapshot(resolve_path(config["source_validation"]))
            or frozen["target_adaptation_files"] != data_snapshot(resolve_path(protocol["target_data"]["adaptation_train"]))):
        raise ValueError("Calibration fitting data changed")

    target_path = evaluation_target(protocol, "development")
    if str(target_path) != frozen["target_development"]:
        raise ValueError("Development split changed")

    before_snapshot = data_snapshot(target_path)
    loader = make_loader(
        target_path,
        provenance["target_dim"],
        protocol["training"]["batch_size"],
    )

    v5b_raw, labels_b = collect_margins(teacher, loader, labeled=True)
    v5d_raw, labels_d = collect_margins(student, loader, labeled=True)

    if not torch.equal(labels_b, labels_d):
        raise ValueError("V5b/V5d development row alignment changed")

    labels = labels_b.cpu().numpy().astype(np.int64)
    v5b_raw = v5b_raw.cpu().numpy().astype(np.float64)
    v5d_raw = v5d_raw.cpu().numpy().astype(np.float64)

    v5b_params = frozen["parameters"]["v5b"]
    v5d_params = frozen["parameters"]["v5d"]
    v5b_scores = np.asarray(
        calibrate_margin(torch.from_numpy(v5b_raw), v5b_params["a"], v5b_params["b"])
    )
    v5d_scores = np.asarray(
        calibrate_margin(torch.from_numpy(v5d_raw), v5d_params["a"], v5d_params["b"])
    )

    v5b_threshold = float(frozen["thresholds"]["v5b"])
    v5d_threshold = float(frozen["thresholds"]["v5d"])

    if (before_snapshot != data_snapshot(target_path)
            or frozen["dependencies"] != dependencies(config, provenance)
            or json.loads(artifact_path.read_text()) != artifact):
        raise ValueError("Development data, dependencies or calibration changed during diagnosis")

    models = {}
    for name, raw, scores, threshold in (
        ("v5b", v5b_raw, v5b_scores, v5b_threshold),
        ("v5d", v5d_raw, v5d_scores, v5d_threshold),
    ):
        current = compute_metrics(labels, scores, threshold)
        models[name] = {
            "current_operating_metrics": current,
            "calibrated_score_summary": _score_summary(labels, scores, threshold),
            "raw_margin_summary": _score_summary(
                labels,
                raw,
                (threshold - frozen["parameters"][name]["b"])
                / frozen["parameters"][name]["a"],
            ),
            "recall_at_target_fpr": _recall_at_target_fpr(labels, scores),
        }

    result = {
        "diagnostic": "V5b vs V5d recall failure analysis",
        "phase": "development_diagnostic_only",
        "warning": (
            "Recall@target-FPR thresholds use development labels and are ORACLE DIAGNOSTICS ONLY; "
            "do not reuse them for deployment or final testing."
        ),
        "training_seed": config["training_seed"],
        "teacher_seed": config["teacher_seed"],
        "target_data": str(target_path),
        "development_files": before_snapshot,
        "calibration_sha256": artifact["sha256"],
        "dependencies": frozen["dependencies"],
        "source_fpr_policy": frozen["max_source_fpr"],
        "models": models,
        "ranking_agreement": _ranking_agreement(v5b_scores, v5d_scores),
        "score_shift_v5d_minus_v5b": _score_shift(labels, v5b_scores, v5d_scores),
        "attack_transition_at_current_thresholds": _attack_transition(
            labels, v5b_scores, v5b_threshold, v5d_scores, v5d_threshold
        ),
        "normal_transition_at_current_thresholds": _normal_transition(
            labels, v5b_scores, v5b_threshold, v5d_scores, v5d_threshold
        ),
    }

    output = (
        resolve_path(args.output)
        if args.output
        else resolve_path(config["result_dir"])
        / f"diagnose_v5d_recall_seed{config['training_seed']}.json"
    )
    if output.exists():
        raise FileExistsError(f"Diagnostic already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")

    _print_current(result)
    _print_recall_table(result)

    print("\n=== Attack transition at current frozen thresholds ===")
    attack = result["attack_transition_at_current_thresholds"]
    for key in ("both_detected", "v5d_recovered_vs_v5b", "v5d_lost_vs_v5b", "both_missed"):
        row = attack[key]
        print(f"{key}: {row['count']} ({row['fraction_of_attacks']:.2%})")
    print(f"net_attack_gain_v5d_minus_v5b: {attack['net_attack_gain_v5d_minus_v5b']}")

    print("\n=== Calibrated Attack score quantiles ===")
    for name in ("v5b", "v5d"):
        q = result["models"][name]["calibrated_score_summary"]["attack_quantiles"]
        print(name.upper() + ": " + ", ".join(f"{k}={v:.6f}" for k, v in q.items()))

    print("\n=== Calibrated Normal upper-tail quantiles ===")
    for name in ("v5b", "v5d"):
        q = result["models"][name]["calibrated_score_summary"]["normal_quantiles"]
        print(name.upper() + ": " + ", ".join(f"{k}={v:.6f}" for k, v in q.items()))

    print("\n=== Ranking agreement ===")
    print(json.dumps(result["ranking_agreement"], indent=2))

    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
