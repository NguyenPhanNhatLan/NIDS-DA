"""Fit separate calibrators without target labels, then compare on development."""
import argparse
import json
import math

import torch

from evaluation.baseline import compute_metrics
from evaluation.calibration import calibrate_margin, fit_affine_calibrator, select_threshold_by_fpr
from evaluation.hda_v5b_calibration import collect_margins, data_snapshot, payload_hash, save_frozen
from training.hda_v5b import file_hash
from training.thesis_protocol import evaluation_target, resolve_path
from training.v5c_common import audit_paths, checkpoint_path, load_audit, load_context, load_student, pipeline_hashes
from training.v6_data import make_loader


def calibration_path(config):
    return resolve_path(config["calibration_dir"]) / f"seed{config['seed']}.json"


def dependencies(config, provenance):
    audit_path, pool_path = audit_paths(config)
    return {"provenance": provenance, "code_sha256": pipeline_hashes(),
            "v5c_checkpoint_sha256": file_hash(checkpoint_path(config)),
            "audit_sha256": file_hash(audit_path), "pool_sha256": file_hash(pool_path)}


def fit(config_path):
    config, protocol, provenance, source, _, teacher = load_context(config_path)
    output = calibration_path(config)
    if output.exists():
        raise FileExistsError(f"Calibration already frozen: {output}")
    audit, pools = load_audit(config, protocol, provenance)
    student = load_student(config, provenance, source)
    deps = dependencies(config, provenance)
    source_path = resolve_path(config["source_validation"])
    snapshot = data_snapshot(source_path)
    batch_size = protocol["training"]["batch_size"]
    source_margins, source_labels = collect_margins(
        source, make_loader(source_path, provenance["source_dim"], batch_size), labeled=True)
    threshold = select_threshold_by_fpr(source_labels.numpy(), source_margins.numpy(), config["calibration_max_fpr"])
    parameters = {}
    for name, model in (("v5b", teacher), ("v5c", student)):
        normal, _ = collect_margins(model, pools["normal"].split(batch_size))
        attack, _ = collect_margins(model, pools["attack"].split(batch_size))
        parameters[name] = fit_affine_calibrator(source_margins, source_labels, normal, attack)
    target_path = resolve_path(protocol["target_data"]["adaptation_train"])
    if (snapshot != data_snapshot(source_path)
            or audit["target_train_files"] != data_snapshot(target_path)
            or deps != dependencies(config, provenance)):
        raise ValueError("Calibration inputs changed during fitting")
    payload = {
        "dependencies": deps, "seed": config["seed"], "parameters": parameters,
        "protocol": "Separate positive affine fits; same filtered train pools and source FPR threshold",
        "source_margin_threshold": threshold, "max_source_fpr": config["calibration_max_fpr"],
        "source_validation": str(source_path), "source_validation_files": snapshot,
        "target_train_files": audit["target_train_files"],
        "target_development": str(evaluation_target(protocol, "development")),
        "target_labels_used_for_fit": False,
    }
    save_frozen(output, payload)
    print(f"Frozen separate V5b/V5c calibrators: {output}")


def report(config_path):
    config, protocol, provenance, source, _, teacher = load_context(config_path)
    output = resolve_path(config["result_dir"]) / f"v5c_vs_v5b_seed{config['seed']}.json"
    if output.exists():
        raise FileExistsError(f"Report already exists: {output}")
    load_audit(config, protocol, provenance)
    student = load_student(config, provenance, source)
    artifact = json.loads(calibration_path(config).read_text())
    frozen = artifact["frozen"]
    if (payload_hash(frozen) != artifact["sha256"]
            or frozen["dependencies"] != dependencies(config, provenance)):
        raise ValueError("Calibration/config/models/code/audit changed; fit a new calibration")
    if (frozen["source_validation_files"] != data_snapshot(resolve_path(config["source_validation"]))
            or frozen["target_train_files"] != data_snapshot(resolve_path(protocol["target_data"]["adaptation_train"]))):
        raise ValueError("Calibration data changed")
    target_path = evaluation_target(protocol, "development")
    if str(target_path) != frozen["target_development"]:
        raise ValueError("Development path changed after calibration")
    snapshot = data_snapshot(target_path)
    metrics, raw_metrics = {}, {}
    previous_labels = None
    for name, model in (("v5b", teacher), ("v5c", student)):
        margins, labels = collect_margins(
            model, make_loader(target_path, provenance["target_dim"], protocol["training"]["batch_size"]), labeled=True)
        if previous_labels is not None and not torch.equal(labels, previous_labels):
            raise ValueError("Development row alignment changed")
        previous_labels = labels
        params = frozen["parameters"][name]
        if not math.isfinite(params["a"]) or params["a"] <= 0 or not math.isfinite(params["b"]):
            raise ValueError("Invalid affine calibrator")
        calibrated = calibrate_margin(margins, params["a"], params["b"])
        metrics[name] = compute_metrics(labels.numpy(), calibrated.numpy(), frozen["source_margin_threshold"])
        raw_metrics[name] = compute_metrics(labels.numpy(), margins.numpy(), provenance["threshold_policy"]["v2_margin_threshold"])
    if snapshot != data_snapshot(target_path):
        raise ValueError("Development data changed during evaluation")
    names = ("pr_auc", "roc_auc", "f1", "recall", "fpr")
    result = {
        "experiment_id": config["experiment_id"], "phase": "development", "seed": config["seed"],
        "calibration_sha256": artifact["sha256"], "dependencies": frozen["dependencies"],
        "target_data": str(target_path), "development_files": snapshot,
        "target_labels_used_for_training_or_calibration": False,
        "pr_auc_definition": "average precision (AP)",
        "comparison_policy": frozen["protocol"],
        "source_margin_threshold": frozen["source_margin_threshold"],
        "calibration_parameters": frozen["parameters"],
        "metrics": metrics, "raw_source_threshold_metrics": raw_metrics,
        "delta_v5c_minus_v5b": {k: metrics["v5c"][k] - metrics["v5b"][k] for k in names},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    for name, row in metrics.items():
        print(name.upper() + " | " + " | ".join(f"{k}={row[k]:.6f}" for k in names))
    print(f"Delta V5c - V5b: {result['delta_v5c_minus_v5b']}")
    print(f"Saved: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hda_v5c.json")
    parser.add_argument("--stage", choices=("fit", "report"), required=True)
    args = parser.parse_args()
    (fit if args.stage == "fit" else report)(args.config)


if __name__ == "__main__":
    main()
