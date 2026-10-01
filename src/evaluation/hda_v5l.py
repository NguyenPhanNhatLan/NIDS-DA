"""Development ranking diagnostic: frozen V5b versus V5l adapter refinement.

Recall@target-FPR thresholds are development-only diagnostics, not deployment
or final-test thresholds. No affine fit, model update or checkpoint mutation.
"""
import argparse
import json

import numpy as np
import pyarrow.parquet as pq
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from evaluation.baseline import compute_metrics
from evaluation.calibration import select_threshold_by_fpr
from evaluation.hda_v5b_calibration import data_snapshot
from training.hda_v5b import file_hash
from training.hda_v5l import checkpoint_path, code_hashes, load_context, load_student
from training.thesis_protocol import evaluation_target, resolve_path
from training.v6_data import make_loader

FPR_BUDGETS = (.01, .02, .03, .05)


def summarize(labels, scores):
    if (labels.shape != scores.shape or not np.isfinite(scores).all()
            or set(np.unique(labels)) != {0, 1}):
        raise ValueError("Need finite aligned margins and both binary classes")
    operating = {}
    for budget in FPR_BUDGETS:
        threshold = select_threshold_by_fpr(labels, scores, budget)
        metrics = compute_metrics(labels, scores, threshold)
        operating[f"fpr_{int(budget*100):02d}pct"] = {
            "target_fpr": budget, "actual_fpr": metrics["fpr"], "recall": metrics["recall"],
            "development_oracle_threshold": threshold,
        }
    return {"pr_auc": float(average_precision_score(labels, scores)),
            "roc_auc": float(roc_auc_score(labels, scores)), "recall_at_target_fpr": operating}


@torch.no_grad()
def report(config_path, output_path=None):
    config, protocol, provenance, source, _, v5b = load_context(config_path)
    output = (resolve_path(output_path) if output_path else
              resolve_path(config["result_dir"]) / f"v5l_vs_v5b_seed{config['training_seed']}.json")
    if output.exists():
        raise FileExistsError(f"Report already exists: {output}")
    student, _ = load_student(config, protocol, provenance, source, v5b)
    models = {"v5b": v5b.cpu().eval(), "v5l": student.cpu().eval()}
    target = evaluation_target(protocol, "development")
    snapshot, code = data_snapshot(target), code_hashes()
    checkpoint_hash = file_hash(checkpoint_path(config))
    expected = sum(pq.read_metadata(p).num_rows for p in sorted(target.glob("*.parquet")))
    scores = {name: [] for name in models}
    labels_parts = []
    for step, (features, labels) in enumerate(make_loader(
            target, provenance["target_dim"], protocol["training"]["batch_size"]), 1):
        labels_parts.append(labels)
        for name, model in models.items():
            _, logits = model(features)
            scores[name].append((logits[:, 1] - logits[:, 0]).double())
        if step % 200 == 0:
            print(f"Scored {step} development batches", flush=True)
    if not labels_parts:
        raise ValueError("Empty development set")
    labels = torch.cat(labels_parts).numpy()
    if len(labels) != expected:
        raise ValueError("Incomplete development scoring")
    metrics = {name: summarize(labels, torch.cat(parts).numpy()) for name, parts in scores.items()}
    if (snapshot != data_snapshot(target) or code != code_hashes()
            or checkpoint_hash != file_hash(checkpoint_path(config))
            or load_context(config_path)[2] != provenance):
        raise ValueError("Evaluation inputs changed during reporting")
    delta = {k: metrics["v5l"][k] - metrics["v5b"][k] for k in ("pr_auc", "roc_auc")}
    delta["recall_at_target_fpr"] = {
        key: metrics["v5l"]["recall_at_target_fpr"][key]["recall"] - row["recall"]
        for key, row in metrics["v5b"]["recall_at_target_fpr"].items()}
    result = {"phase": "development_diagnostic_only", "training_seed": config["training_seed"],
              "provenance": provenance, "code_sha256": code, "checkpoint_sha256": checkpoint_hash,
              "target_data": str(target), "development_files": snapshot, "rows": len(labels),
              "score_space": "raw logit_attack - logit_normal", "metrics": metrics,
              "delta_v5l_minus_v5b": delta,
              "note": "Recall@target-FPR uses development labels; thresholds must not be reused for deployment/final test. No model/calibration fitting. Component-swap diagnostic is not a trained-model reference."}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print("Model | AP | ROC-AUC | R@1% | R@2% | R@3% | R@5%")
    for name, row in metrics.items():
        values = [row["pr_auc"], row["roc_auc"]] + [v["recall"] for v in row["recall_at_target_fpr"].values()]
        print(name.upper() + " | " + " | ".join(f"{v:.6f}" for v in values))
    print("V5l - V5b:", delta)
    print("Saved:", output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hda_v5l.json")
    parser.add_argument("--output")
    args = parser.parse_args()
    report(args.config, args.output)


if __name__ == "__main__":
    main()
