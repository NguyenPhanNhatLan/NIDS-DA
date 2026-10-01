"""Development-only component-swap diagnostic for V5b vs V5d.

This script DOES NOT train, calibrate, or modify checkpoints.

It evaluates four models on the exact same development rows:

1) V5B
   V5b adapter + V5b/source classifier

2) ADAPTER_ONLY_CHANGE
   V5d adapter + V5b/source classifier

3) CLASSIFIER_ONLY_CHANGE
   V5b adapter + V5d classifier

4) FULL_V5D
   V5d adapter + V5d classifier

Primary outputs:
- Average Precision (AP)
- ROC-AUC
- Recall at target FPR budgets: 1%, 2%, 3%, 5%

IMPORTANT:
Recall@target-FPR thresholds are selected using DEVELOPMENT labels.
They are DEV-ORACLE DIAGNOSTICS ONLY and must NOT be reused for
deployment, calibration, or final-test evaluation.
"""

import argparse
import copy
import json
from collections import OrderedDict

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from evaluation.baseline import compute_metrics
from evaluation.calibration import select_threshold_by_fpr
from evaluation.hda_v5b_calibration import data_snapshot, payload_hash
from evaluation.hda_v5d import (
    calibration_path,
    load_v5b_calibration,
    verify_training_data,
    dependencies,
)
from models.hda_v5d import HDAV5DModel
from training.hda_v5d import load_context, load_student
from training.thesis_protocol import evaluation_target, resolve_path
from training.v6_data import make_loader


FPR_BUDGETS = (0.01, 0.02, 0.03, 0.05)


def freeze_eval(model):
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def build_component_models(source, v5b, v5d):
    """Construct the 2x2 adapter/classifier component swap."""
    model_v5b = freeze_eval(copy.deepcopy(v5b))

    adapter_only = HDAV5DModel(
        source_model=source,
        target_adapter=v5d.adapter,
        classifier_only=False,
    )
    adapter_only = freeze_eval(adapter_only)

    classifier_only = HDAV5DModel(
        source_model=source,
        target_adapter=v5b.adapter,
        classifier_only=False,
    )
    classifier_only.classifier.load_state_dict(
        v5d.classifier.state_dict()
    )
    classifier_only = freeze_eval(classifier_only)

    full_v5d = freeze_eval(copy.deepcopy(v5d))

    return OrderedDict([
        ("v5b", model_v5b),
        ("adapter_only_change", adapter_only),
        ("classifier_only_change", classifier_only),
        ("full_v5d", full_v5d),
    ])


def validate_binary(labels):
    labels = np.asarray(labels)
    if labels.ndim != 1 or len(labels) == 0:
        raise ValueError("Labels must be a nonempty 1D vector")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Development labels must be binary")
    if not (labels == 0).any() or not (labels == 1).any():
        raise ValueError("Development set must contain both classes")
    return labels.astype(np.int64)


def validate_scores(name, scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != labels.shape:
        raise ValueError(
            f"{name}: score/label shape mismatch "
            f"{scores.shape} vs {labels.shape}"
        )
    if not np.isfinite(scores).all():
        raise ValueError(f"{name}: scores contain NaN/Inf")
    return scores


def recall_at_target_fpr(labels, scores):
    """DEV ORACLE ONLY."""
    rows = OrderedDict()

    for budget in FPR_BUDGETS:
        threshold = select_threshold_by_fpr(
            labels,
            scores,
            max_fpr=budget,
        )
        metrics = compute_metrics(
            labels,
            scores,
            threshold,
        )

        key = f"fpr_{int(round(100 * budget)):02d}pct"
        rows[key] = {
            "budget": float(budget),
            "oracle_threshold": float(threshold),
            "recall": float(metrics["recall"]),
            "actual_fpr": float(metrics["fpr"]),
            "precision": float(metrics["precision"]),
            "f1": float(metrics["f1"]),
            "tp": int(metrics["tp"]),
            "fp": int(metrics["fp"]),
            "fn": int(metrics["fn"]),
            "tn": int(metrics["tn"]),
        }

    return rows


def ranking_metrics(labels, scores):
    return {
        "pr_auc": float(
            average_precision_score(labels, scores)
        ),
        "roc_auc": float(
            roc_auc_score(labels, scores)
        ),
    }


def score_quantiles(labels, scores):
    normal = scores[labels == 0]
    attack = scores[labels == 1]

    return {
        "normal": {
            "q50": float(np.quantile(normal, 0.50)),
            "q90": float(np.quantile(normal, 0.90)),
            "q95": float(np.quantile(normal, 0.95)),
            "q97": float(np.quantile(normal, 0.97)),
            "q99": float(np.quantile(normal, 0.99)),
        },
        "attack": {
            "q10": float(np.quantile(attack, 0.10)),
            "q25": float(np.quantile(attack, 0.25)),
            "q50": float(np.quantile(attack, 0.50)),
            "q75": float(np.quantile(attack, 0.75)),
            "q90": float(np.quantile(attack, 0.90)),
        },
    }


@torch.no_grad()
def collect_aligned_scores(models, loader):
    labels_parts = []
    scores = OrderedDict((name, []) for name in models)
    for batch_index, (features, labels) in enumerate(loader, 1):
        labels_parts.append(labels.cpu())
        for name, model in models.items():
            _, logits = model(features.to(next(model.parameters()).device))
            scores[name].append((logits[:, 1] - logits[:, 0]).cpu().double())
        if batch_index % 200 == 0:
            print(f"Scored {batch_index} development batches with all four models", flush=True)
    if not labels_parts:
        raise ValueError("Empty development loader")
    return torch.cat(labels_parts).numpy(), OrderedDict(
        (name, torch.cat(parts).numpy()) for name, parts in scores.items())


def delta_vs_v5b(rows):
    base = rows["v5b"]
    out = OrderedDict()

    for name, row in rows.items():
        if name == "v5b":
            continue

        budgets = OrderedDict()
        for budget in FPR_BUDGETS:
            key = f"fpr_{int(round(100 * budget)):02d}pct"
            budgets[key] = (
                row["recall_at_target_fpr"][key]["recall"]
                - base["recall_at_target_fpr"][key]["recall"]
            )

        out[name] = {
            "pr_auc": row["ranking"]["pr_auc"] - base["ranking"]["pr_auc"],
            "roc_auc": row["ranking"]["roc_auc"] - base["ranking"]["roc_auc"],
            "recall_at_target_fpr": budgets,
        }

    return out


def print_main_table(rows):
    print("\n=== Component swap: ranking + Recall@target-FPR ===")
    print("DEV ORACLE ONLY for Recall@target-FPR thresholds.")
    print(
        "Model | AP | ROC-AUC | "
        "R@1%FPR | R@2%FPR | R@3%FPR | R@5%FPR"
    )

    for name, row in rows.items():
        recalls = []
        for budget in FPR_BUDGETS:
            key = f"fpr_{int(round(100 * budget)):02d}pct"
            recalls.append(
                row["recall_at_target_fpr"][key]["recall"]
            )

        print(
            f"{name.upper()} | "
            f"{row['ranking']['pr_auc']:.6f} | "
            f"{row['ranking']['roc_auc']:.6f} | "
            + " | ".join(f"{x:.6f}" for x in recalls)
        )


def print_delta_table(deltas):
    print("\n=== Delta vs V5B ===")
    print(
        "Model | dAP | dROC | "
        "dR@1% | dR@2% | dR@3% | dR@5%"
    )

    for name, row in deltas.items():
        recall_delta = row["recall_at_target_fpr"]
        print(
            f"{name.upper()} | "
            f"{row['pr_auc']:+.6f} | "
            f"{row['roc_auc']:+.6f} | "
            f"{recall_delta['fpr_01pct']:+.6f} | "
            f"{recall_delta['fpr_02pct']:+.6f} | "
            f"{recall_delta['fpr_03pct']:+.6f} | "
            f"{recall_delta['fpr_05pct']:+.6f}"
        )


def print_component_interpretation(rows):
    b = rows["v5b"]
    a = rows["adapter_only_change"]
    c = rows["classifier_only_change"]
    d = rows["full_v5d"]

    print("\n=== Direct component effects relative to V5B ===")

    for label, row in (
        ("Adapter change only", a),
        ("Classifier change only", c),
        ("Full V5D", d),
    ):
        print(
            f"{label}: "
            f"dAP={row['ranking']['pr_auc'] - b['ranking']['pr_auc']:+.6f}, "
            f"dROC={row['ranking']['roc_auc'] - b['ranking']['roc_auc']:+.6f}"
        )

    print(
        "\nUse the table above to identify which isolated component "
        "reproduces the V5D Recall@FPR pattern."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/hda_v5d.json",
    )
    parser.add_argument(
        "--training-seed",
        type=int,
        choices=(42, 43, 44),
        default=42,
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional JSON path. Default: "
            "<result_dir>/diagnose_v5d_components_seedN.json"
        ),
    )
    args = parser.parse_args()

    (
        config,
        protocol,
        provenance,
        source,
        _v2,
        v5b,
    ) = load_context(
        args.config,
        args.training_seed,
    )
    output = (resolve_path(args.output) if args.output is not None else
              resolve_path(config["result_dir"]) / f"diagnose_v5d_components_seed{config['training_seed']}.json")
    if output.exists():
        raise FileExistsError(f"Diagnostic already exists: {output}")

    v5d, checkpoint = load_student(
        config,
        provenance,
        source,
        v5b,
    )

    verify_training_data(
        protocol,
        checkpoint,
    )

    v5b_artifact = load_v5b_calibration(
        config,
        protocol,
        provenance,
    )

    cal_path = calibration_path(config)
    cal_artifact = json.loads(
        cal_path.read_text()
    )
    cal_frozen = cal_artifact["frozen"]
    if (payload_hash(cal_frozen) != cal_artifact["sha256"]
            or cal_frozen["dependencies"] != dependencies(config, provenance)
            or cal_frozen["v5b_calibration_payload_sha256"] != v5b_artifact["sha256"]
            or cal_frozen["source_validation_files"] != data_snapshot(resolve_path(config["source_validation"]))
            or cal_frozen["target_adaptation_files"] != data_snapshot(resolve_path(protocol["target_data"]["adaptation_train"]))):
        raise ValueError("Frozen calibration, dependencies or fitting data changed")

    if cal_frozen["training_seed"] != config["training_seed"]:
        raise ValueError(
            "V5d calibration training seed mismatch"
        )
    if cal_frozen["teacher_seed"] != config["teacher_seed"]:
        raise ValueError(
            "V5d calibration teacher seed mismatch"
        )

    target_path = evaluation_target(
        protocol,
        "development",
    )
    if (
        resolve_path(cal_frozen["target_development"]).resolve()
        != target_path.resolve()
    ):
        raise ValueError(
            "Development split differs from frozen V5d calibration"
        )

    snapshot_before = data_snapshot(
        target_path
    )

    loader = make_loader(
        target_path,
        provenance["target_dim"],
        protocol["training"]["batch_size"],
    )

    models = build_component_models(
        source,
        v5b,
        v5d,
    )

    labels, score_map = collect_aligned_scores(
        models,
        loader,
    )
    labels = validate_binary(labels)

    rows = OrderedDict()

    for name, scores in score_map.items():
        scores = validate_scores(
            name,
            scores,
            labels,
        )

        rows[name] = {
            "components": {
                "adapter": (
                    "v5b"
                    if name in ("v5b", "classifier_only_change")
                    else "v5d"
                ),
                "classifier": (
                    "v5b_source"
                    if name in ("v5b", "adapter_only_change")
                    else "v5d"
                ),
            },
            "ranking": ranking_metrics(
                labels,
                scores,
            ),
            "recall_at_target_fpr": recall_at_target_fpr(
                labels,
                scores,
            ),
            "score_quantiles": score_quantiles(
                labels,
                scores,
            ),
        }

    if (snapshot_before != data_snapshot(target_path)
            or cal_frozen["dependencies"] != dependencies(config, provenance)
            or json.loads(cal_path.read_text()) != cal_artifact):
        raise ValueError(
            "Development data changed during diagnosis"
        )

    deltas = delta_vs_v5b(rows)

    result = {
        "diagnostic": (
            "V5d 2x2 component swap: adapter vs classifier"
        ),
        "phase": "development_diagnostic_only",
        "warning": (
            "Recall@target-FPR thresholds use development labels and are "
            "DEV-ORACLE DIAGNOSTICS ONLY. Do not use them for deployment, "
            "calibration, model selection on final test, or final-test thresholds."
        ),
        "training_seed": int(config["training_seed"]),
        "teacher_seed": int(config["teacher_seed"]),
        "target_data": str(target_path),
        "development_files": snapshot_before,
        "v5d_calibration": str(cal_path),
        "calibration_sha256": cal_artifact["sha256"],
        "dependencies": cal_frozen["dependencies"],
        "score_space": (
            "raw logit margin = attack logit - normal logit; "
            "positive affine calibration omitted because AP/ROC and "
            "Recall@fixed-target-FPR are ranking invariant"
        ),
        "models": rows,
        "delta_vs_v5b": deltas,
    }

    output = (
        resolve_path(args.output)
        if args.output is not None
        else resolve_path(config["result_dir"])
        / f"diagnose_v5d_components_seed{config['training_seed']}.json"
    )

    if output.exists():
        raise FileExistsError(
            f"Diagnostic already exists: {output}"
        )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output.open("x") as stream:
        json.dump(
            result,
            stream,
            indent=2,
            allow_nan=False,
        )
        stream.write("\n")

    print_main_table(rows)
    print_delta_table(deltas)
    print_component_interpretation(rows)

    print("\n=== Model definitions ===")
    print("V5B                    = V5B adapter + V5B/source classifier")
    print("ADAPTER_ONLY_CHANGE    = V5D adapter + V5B/source classifier")
    print("CLASSIFIER_ONLY_CHANGE = V5B adapter + V5D classifier")
    print("FULL_V5D               = V5D adapter + V5D classifier")

    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
