"""Frozen V5b reference and independently fitted V5n affine calibration."""
import argparse
import json
import math

import torch

from evaluation.baseline import compute_metrics
from evaluation.calibration import calibrate_margin, fit_affine_calibrator, select_threshold_by_fpr
from evaluation.hda_v5b_calibration import (
    build_frozen_target_pools, calibration_code_hashes, collect_margins,
    data_snapshot, payload_hash, save_frozen,
)
from training.hda_v5b import ROOT, file_hash
from training.hda_v5n import checkpoint_path, code_hashes, load_context, load_student
from training.thesis_protocol import evaluation_target, resolve_path
from training.v6_data import make_loader


def calibration_path(config):
    return resolve_path(config["calibration_dir"]) / f"seed{config['training_seed']}.json"


def dependencies(config, provenance):
    return {"provenance": provenance, "code_sha256": code_hashes(),
            "v5n_checkpoint_sha256": file_hash(checkpoint_path(config)),
            "v5b_calibration_file_sha256": file_hash(resolve_path(config["v5b_calibration"]))}


def validate_affine(params):
    if (not math.isfinite(params["a"]) or params["a"] <= 0
            or not math.isfinite(params["b"])):
        raise ValueError("Affine calibration requires finite a > 0 and finite b")


def validate_v5b_code_hashes(frozen_hashes, current_hashes):
    """Allow only the reviewed pool-validation refactor when consuming Stage 1.

    Git 0e1cf842 -> 815daced adds build_frozen_target_pools and redirects fit
    through it. Model loading, scoring, affine math and threshold selection are
    unchanged. This exception is for reading the pinned artifact, not refitting.
    """
    reviewed = {
        "src/evaluation/hda_v5b_calibration.py": (
            "276158843ee2b875dd984f75841f75e6b72f1ed169f689f0f3e4be7db8f38651",
            "af7de97279b5e48fbcbbfc0961cf807e4d66ae6ce9728e2f2fadfafda266b6db",
        )
    }
    accepted = []
    for name in sorted(set(frozen_hashes) | set(current_hashes)):
        expected, actual = frozen_hashes.get(name), current_hashes.get(name)
        if expected == actual:
            continue
        if name in reviewed and (expected, actual) == reviewed[name]:
            accepted.append(name)
            continue
        raise ValueError(f"V5b calibration code changed: {name}; frozen={expected}; current={actual}")
    return accepted


def load_v5b_calibration(config, protocol, provenance):
    """Read the pinned Stage 1 artifact; never refit or overwrite it."""
    artifact = json.loads(resolve_path(config["v5b_calibration"]).read_text())
    frozen = artifact["frozen"]
    if (payload_hash(frozen) != artifact["sha256"]
            or artifact["sha256"] != config["v5b_calibration_payload_sha256"]):
        raise ValueError("V5b calibration does not match the pinned frozen artifact")
    if frozen["checkpoint_sha256"] != provenance["stage1_checkpoint_sha256"]:
        raise ValueError("V5b calibration checkpoint mismatch")
    expected_dependencies = {key: provenance[key] for key in (
        "source_seed", "source_dim", "target_dim", "source_checkpoint_sha256", "teacher_checkpoint_sha256")}
    if frozen["model_dependencies"] != expected_dependencies:
        raise ValueError("V5b calibration model dependencies mismatch")
    accepted = validate_v5b_code_hashes(frozen["code_sha256"], calibration_code_hashes())
    reference_path = resolve_path(frozen["reference"])
    current_reference_hash = file_hash(reference_path)
    if current_reference_hash != frozen["reference_sha256"]:
        raise ValueError(f"V5b calibration reference changed: {reference_path}; "
                         f"frozen={frozen['reference_sha256']}; current={current_reference_hash}")
    if frozen["threshold_policy"]["max_fpr"] != config["calibration_max_fpr"]:
        raise ValueError("V5b and V5n must use the same source FPR policy")
    if frozen["target_labels_used_for_fit"] is not False:
        raise ValueError("V5b reference must not use target labels for calibration")
    validate_affine(frozen)
    if not math.isfinite(frozen["source_margin_threshold"]):
        raise ValueError("Invalid V5b source threshold")
    source_path = resolve_path(config["source_validation"])
    target_path = resolve_path(protocol["target_data"]["adaptation_train"])
    for key, expected_path in (("source_validation", source_path),
                               ("target_adaptation_train", target_path),
                               ("target_development", evaluation_target(protocol, "development"))):
        if resolve_path(frozen[key]).resolve() != expected_path.resolve():
            raise ValueError(f"V5b reference split changed: {key}")
    for key, path in (("source_validation_files", source_path), ("target_adaptation_files", target_path)):
        current = data_snapshot(path)
        expected = frozen[key]
        changed = [name for name in sorted(set(expected) | set(current))
                   if expected.get(name) != current.get(name)]
        if changed:
            raise ValueError(f"V5b calibration data changed in {path}: {', '.join(changed)}")
    if accepted:
        print("Stage 1 compatibility: reviewed pool-validation refactor accepted: "
              + ", ".join(accepted), flush=True)
    return artifact


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
        raise ValueError("Training data changed since V5n checkpoint")


def operating_metrics(labels, margins, parameters, threshold):
    validate_affine(parameters)
    calibrated = calibrate_margin(margins, parameters["a"], parameters["b"])
    return {
        "raw": compute_metrics(labels.numpy(), margins.numpy(), threshold),
        "calibrated": compute_metrics(labels.numpy(), calibrated.numpy(), threshold),
        "raw_equivalent_calibrated_threshold": (threshold - parameters["b"]) / parameters["a"],
    }


def fit(config_path, training_seed=None):
    config, protocol, provenance, source, v2, teacher = load_context(config_path, training_seed)
    output = calibration_path(config)
    if output.exists():
        raise FileExistsError(f"Calibration already exists: {output}")
    student, checkpoint = load_student(config, provenance, source, teacher)
    verify_training_data(protocol, checkpoint)
    v5b_artifact = load_v5b_calibration(config, protocol, provenance)
    v5b = v5b_artifact["frozen"]
    deps = dependencies(config, provenance)
    source_path = resolve_path(config["source_validation"])
    target_path = resolve_path(protocol["target_data"]["adaptation_train"])
    source_snapshot, target_snapshot = data_snapshot(source_path), data_snapshot(target_path)
    batch_size = protocol["training"]["batch_size"]
    margins, labels = collect_source_margins(
        student, make_loader(source_path, provenance["source_dim"], batch_size))
    threshold = select_threshold_by_fpr(labels.numpy(), margins.numpy(), config["calibration_max_fpr"])
    # Only V2 quantile candidates from unlabeled adaptation train, same frozen policy as V5b.
    normal, attack, pseudo = build_frozen_target_pools(v2, protocol, provenance["target_dim"], batch_size)
    normal_margins, _ = collect_margins(student, normal.split(batch_size))
    attack_margins, _ = collect_margins(student, attack.split(batch_size))
    params = fit_affine_calibrator(
        source_margin=margins, source_label=labels,
        target_margin_normal=normal_margins, target_margin_attack=attack_margins)
    validate_affine(params)
    if (source_snapshot != data_snapshot(source_path) or target_snapshot != data_snapshot(target_path)
            or deps != dependencies(config, provenance)):
        raise ValueError("Calibration inputs changed while fitting")
    save_frozen(output, {
        "dependencies": deps, "training_seed": config["training_seed"],
        "deployment_classifier": "original_source", "ema_decay": checkpoint["ema_decay"],
        "bn_recalibrated": checkpoint["bn_recalibrated"],
        "teacher_seed": config["teacher_seed"], "classifier_only": config["classifier_only"],
        "thresholds": {"v5b": v5b["source_margin_threshold"], "v5n": threshold},
        "parameters": {"v5b": {"a": v5b["a"], "b": v5b["b"]}, "v5n": params},
        "v5b_calibration_payload_sha256": v5b_artifact["sha256"],
        "protocol": "Frozen V5b affine artifact; separate V5n affine fit on V2 train pools; same source FPR policy",
        "max_source_fpr": config["calibration_max_fpr"],
        "source_metrics": {"v5b": v5b["source_validation_metrics"],
                           "v5n": compute_metrics(labels.numpy(), margins.numpy(), threshold)},
        "source_validation": str(source_path), "source_validation_files": source_snapshot,
        "target_adaptation_train": str(target_path), "target_adaptation_files": target_snapshot,
        "pseudo_metadata": pseudo,
        "target_development": str(evaluation_target(protocol, "development")),
        "target_labels_used_for_fit": False,
    })
    print(f"Frozen V5n affine a={params['a']:.8f}, b={params['b']:.8f}; source threshold={threshold:.8f}")
    print(f"Reused V5b frozen calibration: {config['v5b_calibration']}")
    print(f"Saved: {output}")


def report(config_path, training_seed=None):
    config, protocol, provenance, source, _, teacher = load_context(config_path, training_seed)
    output = resolve_path(config["result_dir"]) / f"v5n_vs_v5b_seed{config['training_seed']}.json"
    if output.exists():
        raise FileExistsError(f"Report already exists: {output}")
    student, checkpoint = load_student(config, provenance, source, teacher)
    verify_training_data(protocol, checkpoint)
    v5b_artifact = load_v5b_calibration(config, protocol, provenance)
    artifact = json.loads(calibration_path(config).read_text())
    frozen = artifact["frozen"]
    if (payload_hash(frozen) != artifact["sha256"]
            or frozen["dependencies"] != dependencies(config, provenance)
            or frozen["v5b_calibration_payload_sha256"] != v5b_artifact["sha256"]
            or frozen["source_validation_files"] != data_snapshot(resolve_path(config["source_validation"]))
            or frozen["target_adaptation_files"] != data_snapshot(resolve_path(protocol["target_data"]["adaptation_train"]))):
        raise ValueError("Frozen calibration, dependencies or fitting data changed")
    target_path = evaluation_target(protocol, "development")
    if str(target_path) != frozen["target_development"]:
        raise ValueError("Development path changed")
    snapshot = data_snapshot(target_path)
    loader = make_loader(target_path, provenance["target_dim"], protocol["training"]["batch_size"])
    operating = {}
    previous_labels = None
    for name, model in (("v5b", teacher), ("v5n", student)):
        margins, labels = collect_margins(model, loader, labeled=True)
        if previous_labels is not None and not torch.equal(labels, previous_labels):
            raise ValueError("Development row alignment changed")
        previous_labels = labels
        operating[name] = operating_metrics(labels, margins, frozen["parameters"][name], frozen["thresholds"][name])
    if snapshot != data_snapshot(target_path) or frozen["dependencies"] != dependencies(config, provenance):
        raise ValueError("Evaluation inputs changed during reporting")
    names = ("pr_auc", "roc_auc", "f1", "recall", "fpr")
    metrics = {name: row["calibrated"] for name, row in operating.items()}
    result = {
        "experiment_id": config["experiment_id"], "training_seed": config["training_seed"],
        "teacher_seed": config["teacher_seed"], "classifier_only": config["classifier_only"], "phase": "development",
        "target_data": str(target_path), "development_files": snapshot,
        "calibration_sha256": artifact["sha256"], "dependencies": frozen["dependencies"],
        "v5b_calibration_payload_sha256": v5b_artifact["sha256"],
        "source_thresholds": frozen["thresholds"], "parameters": frozen["parameters"],
        "max_source_fpr": frozen["max_source_fpr"], "comparison_policy": frozen["protocol"],
        "pr_auc_definition": "average precision (AP)",
        "target_labels_used_for_training_or_calibration": False,
        "metrics": metrics, "operating_points": operating,
        "raw_definition": "Uncalibrated target margin at that model's source threshold",
        "calibration_limitation": "Noisy pseudo anchors; positive affine preserves ranking, not an AP improvement",
        "delta_v5n_minus_v5b": {k: metrics["v5n"][k] - metrics["v5b"][k] for k in names},
        "raw_delta_v5n_minus_v5b": {k: operating["v5n"]["raw"][k] - operating["v5b"]["raw"][k] for k in names},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    for name, row in operating.items():
        for kind in ("raw", "calibrated"):
            print(name.upper() + f" {kind} | " + " | ".join(f"{k}={row[kind][k]:.6f}" for k in names))
    print(f"Calibrated delta V5n - V5b: {result['delta_v5n_minus_v5b']}")
    print(f"Saved: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hda_v5n.json")
    parser.add_argument("--training-seed", type=int, choices=(42, 43, 44))
    parser.add_argument("--stage", choices=("fit", "report"), required=True)
    args = parser.parse_args()
    (fit if args.stage == "fit" else report)(args.config, args.training_seed)


if __name__ == "__main__":
    main()
