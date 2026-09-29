"""Freeze per-model source-validation thresholds, then compare on development."""
import argparse
import json

import torch

from evaluation.baseline import compute_metrics
from evaluation.calibration import select_threshold_by_fpr
from evaluation.hda_v5b_calibration import collect_margins, data_snapshot, payload_hash, save_frozen
from training.hda_v5b import ROOT, file_hash
from training.hda_v5d import checkpoint_path, code_hashes, load_context, load_student
from training.thesis_protocol import evaluation_target, resolve_path
from training.v6_data import make_loader


def calibration_path(config):
    return resolve_path(config["calibration_dir"]) / f"seed{config['seed']}.json"


def dependencies(config, provenance):
    return {"provenance": provenance, "code_sha256": code_hashes(),
            "v5d_checkpoint_sha256": file_hash(checkpoint_path(config))}


def collect_source_margins(student, loader):
    student.eval()
    margins, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            _, logits = student.forward_source(x)
            margins.append((logits[:, 1] - logits[:, 0]).cpu().double())
            labels.append(y.cpu())
    if not margins:
        raise ValueError("Empty source validation")
    return torch.cat(margins), torch.cat(labels)


def verify_training_data(protocol, checkpoint):
    actual = {"source": data_snapshot(ROOT / "data/features/unsw_train"),
              "target": data_snapshot(resolve_path(protocol["target_data"]["adaptation_train"]))}
    if actual != checkpoint["training_data"]:
        raise ValueError("Training data changed since V5d checkpoint")


def fit(config_path):
    config, protocol, provenance, source, _, teacher = load_context(config_path)
    output = calibration_path(config)
    if output.exists():
        raise FileExistsError(f"Calibration already exists: {output}")
    student, checkpoint = load_student(config, provenance, source, teacher)
    verify_training_data(protocol, checkpoint)
    deps = dependencies(config, provenance)
    source_path = resolve_path(config["source_validation"])
    snapshot = data_snapshot(source_path)
    loader = make_loader(source_path, provenance["source_dim"], protocol["training"]["batch_size"])
    v5b_margins, labels = collect_margins(source, loader, labeled=True)
    v5d_margins, v5d_labels = collect_source_margins(student, loader)
    if not torch.equal(labels, v5d_labels):
        raise ValueError("Source validation row alignment changed")
    thresholds, metrics = {}, {}
    for name, margins in (("v5b", v5b_margins), ("v5d", v5d_margins)):
        threshold = select_threshold_by_fpr(labels.numpy(), margins.numpy(), config["calibration_max_fpr"])
        thresholds[name] = threshold
        metrics[name] = compute_metrics(labels.numpy(), margins.numpy(), threshold)
    if snapshot != data_snapshot(source_path) or deps != dependencies(config, provenance):
        raise ValueError("Calibration inputs changed while fitting")
    save_frozen(output, {
        "dependencies": deps, "thresholds": thresholds,
        "protocol": "Per-model UNSW validation margin threshold at the same maximum source FPR; no affine fit",
        "max_source_fpr": config["calibration_max_fpr"], "source_metrics": metrics,
        "source_validation": str(source_path), "source_validation_files": snapshot,
        "target_development": str(evaluation_target(protocol, "development")),
        "target_labels_used_for_fit": False,
    })
    print(f"Frozen thresholds: {thresholds}")
    print(f"Saved: {output}")


def report(config_path):
    config, protocol, provenance, source, _, teacher = load_context(config_path)
    output = resolve_path(config["result_dir"]) / f"v5d_vs_v5b_seed{config['seed']}.json"
    if output.exists():
        raise FileExistsError(f"Report already exists: {output}")
    student, checkpoint = load_student(config, provenance, source, teacher)
    verify_training_data(protocol, checkpoint)
    artifact = json.loads(calibration_path(config).read_text())
    frozen = artifact["frozen"]
    if (payload_hash(frozen) != artifact["sha256"]
            or frozen["dependencies"] != dependencies(config, provenance)
            or frozen["source_validation_files"] != data_snapshot(resolve_path(config["source_validation"]))):
        raise ValueError("Frozen calibration, dependencies or source validation changed")
    target_path = evaluation_target(protocol, "development")
    if str(target_path) != frozen["target_development"]:
        raise ValueError("Development path changed")
    snapshot = data_snapshot(target_path)
    loader = make_loader(target_path, provenance["target_dim"], protocol["training"]["batch_size"])
    metrics = {}
    previous_labels = None
    for name, model in (("v5b", teacher), ("v5d", student)):
        margins, labels = collect_margins(model, loader, labeled=True)
        if previous_labels is not None and not torch.equal(labels, previous_labels):
            raise ValueError("Development row alignment changed")
        previous_labels = labels
        metrics[name] = compute_metrics(labels.numpy(), margins.numpy(), frozen["thresholds"][name])
    if snapshot != data_snapshot(target_path) or frozen["dependencies"] != dependencies(config, provenance):
        raise ValueError("Evaluation inputs changed during reporting")
    names = ("pr_auc", "roc_auc", "f1", "recall", "fpr")
    result = {
        "experiment_id": config["experiment_id"], "seed": config["seed"], "phase": "development",
        "target_data": str(target_path), "development_files": snapshot,
        "calibration_sha256": artifact["sha256"], "dependencies": frozen["dependencies"],
        "thresholds": frozen["thresholds"], "comparison_policy": frozen["protocol"],
        "pr_auc_definition": "average precision (AP)", "score_space": "attack minus normal logit",
        "target_labels_used_for_training_or_calibration": False,
        "metrics": metrics,
        "delta_v5d_minus_v5b": {k: metrics["v5d"][k] - metrics["v5b"][k] for k in names},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    for name, row in metrics.items():
        print(name.upper() + " | " + " | ".join(f"{k}={row[k]:.6f}" for k in names))
    print(f"Delta V5d - V5b: {result['delta_v5d_minus_v5b']}")
    print(f"Saved: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hda_v5d.json")
    parser.add_argument("--stage", choices=("fit", "report"), required=True)
    args = parser.parse_args()
    (fit if args.stage == "fit" else report)(args.config)


if __name__ == "__main__":
    main()
