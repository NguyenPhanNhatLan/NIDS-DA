"""V5g calibration/report with frozen V5b/V5d/V5f references."""
import argparse
import json
import math

import torch

from evaluation.baseline import compute_metrics
from evaluation.calibration import calibrate_margin, fit_affine_calibrator, select_threshold_by_fpr
from evaluation.hda_v5b_calibration import (
    build_frozen_target_pools, collect_margins, data_snapshot, payload_hash, save_frozen,
)
from evaluation.hda_v5f import (
    collect_source_margins, load_v5b_calibration, operating_metrics, validate_affine,
    verify_training_data,
)
from training.hda_v5b import file_hash
from training.hda_v5f import load_v5d_reference
from training.hda_v5g import checkpoint_path, code_hashes, load_context, load_student
from training.thesis_protocol import evaluation_target, resolve_path
from training.v6_data import make_loader


def calibration_path(config):
    return resolve_path(config["calibration_dir"]) / f"seed{config['training_seed']}.json"


def dependencies(config, provenance):
    return {
        "provenance": provenance,
        "code_sha256": code_hashes(),
        "v5g_checkpoint_sha256": file_hash(checkpoint_path(config)),
        "v5b_calibration_file_sha256": file_hash(resolve_path(config["v5b_calibration"])),
        "v5d_reference_sha256": file_hash(resolve_path(config["v5d_reference_report"])),
        "v5d_reference_checkpoint_sha256": file_hash(resolve_path(config["v5d_reference_checkpoint"])),
    }


def load_v5f_reference(config, protocol):
    path = resolve_path(config["v5f_reference_report"])
    report = json.loads(path.read_text())
    if (report.get("phase") != "development"
            or report.get("training_seed") != config["training_seed"]
            or resolve_path(report["target_data"]).resolve() != evaluation_target(protocol, "development").resolve()
            or "v5f" not in report.get("metrics", {})):
        raise ValueError("V5f reference report is incompatible with V5g comparison")
    return report, file_hash(path)


def fit(config_path, training_seed=None):
    config, protocol, provenance, source, v2, teacher = load_context(config_path, training_seed)
    load_v5d_reference(config, protocol, provenance)
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
    threshold = select_threshold_by_fpr(
        labels.numpy(), margins.numpy(), config["calibration_max_fpr"])

    # Keep calibration identical in principle to V5f: frozen V2 hard anchors only.
    normal, attack, pseudo = build_frozen_target_pools(
        v2, protocol, provenance["target_dim"], batch_size)
    normal_margins, _ = collect_margins(student, normal.split(batch_size))
    attack_margins, _ = collect_margins(student, attack.split(batch_size))
    params = fit_affine_calibrator(
        source_margin=margins,
        source_label=labels,
        target_margin_normal=normal_margins,
        target_margin_attack=attack_margins,
    )
    validate_affine(params)

    if (source_snapshot != data_snapshot(source_path)
            or target_snapshot != data_snapshot(target_path)
            or deps != dependencies(config, provenance)):
        raise ValueError("Calibration inputs changed while fitting")

    save_frozen(output, {
        "dependencies": deps,
        "training_seed": config["training_seed"],
        "teacher_seed": config["teacher_seed"],
        "classifier_only": config["classifier_only"],
        "thresholds": {"v5b": v5b["source_margin_threshold"], "v5g": threshold},
        "parameters": {"v5b": {"a": v5b["a"], "b": v5b["b"]}, "v5g": params},
        "v5b_calibration_payload_sha256": v5b_artifact["sha256"],
        "protocol": "Frozen V5b affine artifact; separate V5g affine fit on same V2 hard anchors; same source FPR policy",
        "max_source_fpr": config["calibration_max_fpr"],
        "source_metrics": {
            "v5b": v5b["source_validation_metrics"],
            "v5g": compute_metrics(labels.numpy(), margins.numpy(), threshold),
        },
        "source_validation": str(source_path),
        "source_validation_files": source_snapshot,
        "target_adaptation_train": str(target_path),
        "target_adaptation_files": target_snapshot,
        "pseudo_metadata": pseudo,
        "target_development": str(evaluation_target(protocol, "development")),
        "target_labels_used_for_fit": False,
    })
    print(f"Frozen V5g affine a={params['a']:.8f}, b={params['b']:.8f}; source threshold={threshold:.8f}")
    print(f"Saved: {output}")


def report(config_path, training_seed=None):
    config, protocol, provenance, source, _, teacher = load_context(config_path, training_seed)
    v5d_reference = load_v5d_reference(config, protocol, provenance)
    v5f_reference, v5f_reference_sha = load_v5f_reference(config, protocol)
    output = resolve_path(config["result_dir"]) / f"v5g_vs_v5f_seed{config['training_seed']}.json"
    if output.exists():
        raise FileExistsError(f"Report already exists: {output}")

    student, checkpoint = load_student(config, provenance, source, teacher)
    verify_training_data(protocol, checkpoint)
    v5b_artifact = load_v5b_calibration(config, protocol, provenance)
    artifact = json.loads(calibration_path(config).read_text())
    frozen = artifact["frozen"]
    deps = dependencies(config, provenance)
    if (payload_hash(frozen) != artifact["sha256"]
            or frozen["dependencies"] != deps
            or frozen["v5b_calibration_payload_sha256"] != v5b_artifact["sha256"]
            or frozen["source_validation_files"] != data_snapshot(resolve_path(config["source_validation"]))
            or frozen["target_adaptation_files"] != data_snapshot(resolve_path(protocol["target_data"]["adaptation_train"]))):
        raise ValueError("Frozen calibration, dependencies or fitting data changed")

    target_path = evaluation_target(protocol, "development")
    if str(target_path) != frozen["target_development"]:
        raise ValueError("Development path changed")
    snapshot = data_snapshot(target_path)
    if snapshot != v5d_reference["development_files"]:
        raise ValueError("V5d/V5g development snapshots differ")
    if snapshot != v5f_reference["development_files"]:
        raise ValueError("V5f/V5g development snapshots differ")

    loader = make_loader(target_path, provenance["target_dim"], protocol["training"]["batch_size"])
    operating = {}
    previous_labels = None
    # V5b reference model is the frozen V5b teacher returned by load_context.
    v5b_params = v5b_artifact["frozen"]
    pairs = (
        ("v5b", teacher, {"a": v5b_params["a"], "b": v5b_params["b"]},
         v5b_params["source_margin_threshold"]),
        ("v5g", student, frozen["parameters"]["v5g"], frozen["thresholds"]["v5g"]),
    )
    for name, model, params, threshold in pairs:
        margins, labels = collect_margins(model, loader, labeled=True)
        if previous_labels is not None and not torch.equal(labels, previous_labels):
            raise ValueError("Development row alignment changed")
        previous_labels = labels
        operating[name] = operating_metrics(labels, margins, params, threshold)

    if (snapshot != data_snapshot(target_path)
            or deps != dependencies(config, provenance)
            or v5f_reference_sha != file_hash(resolve_path(config["v5f_reference_report"]))):
        raise ValueError("Evaluation inputs changed during reporting")

    names = ("pr_auc", "roc_auc", "f1", "recall", "fpr")
    metrics = {
        "v5b": operating["v5b"]["calibrated"],
        "v5d": v5d_reference["metrics"]["v5d"],
        "v5f": v5f_reference["metrics"]["v5f"],
        "v5g": operating["v5g"]["calibrated"],
    }
    result = {
        "experiment_id": config["experiment_id"],
        "training_seed": config["training_seed"],
        "teacher_seed": config["teacher_seed"],
        "classifier_only": config["classifier_only"],
        "phase": "development",
        "target_data": str(target_path),
        "development_files": snapshot,
        "calibration_sha256": artifact["sha256"],
        "dependencies": deps,
        "v5f_reference_report": config["v5f_reference_report"],
        "v5f_reference_sha256": v5f_reference_sha,
        "source_thresholds": frozen["thresholds"],
        "parameters": frozen["parameters"],
        "max_source_fpr": frozen["max_source_fpr"],
        "pr_auc_definition": "average precision (AP)",
        "target_labels_used_for_training_or_calibration": False,
        "metrics": metrics,
        "operating_points": operating,
        "delta_v5g_minus_v5f": {k: metrics["v5g"][k] - metrics["v5f"][k] for k in names},
        "delta_v5g_minus_v5d": {k: metrics["v5g"][k] - metrics["v5d"][k] for k in names},
        "delta_v5g_minus_v5b": {k: metrics["v5g"][k] - metrics["v5b"][k] for k in names},
        "fixed_hard_anchor_diagnostics": {
            "v5b_initial": checkpoint["fixed_diagnostic_v5b_initial"],
            "v5d_reference": checkpoint["fixed_diagnostic_v5d_reference"],
            "v5g_final": checkpoint["history"][-1]["fixed_hard_anchor_diagnostic"],
        },
        "conditional_alignment": checkpoint["conditional_alignment"],
        "calibration_limitation": "Positive affine calibration preserves ranking/AP; V2 pseudo anchors remain noisy.",
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")

    print("Model | AP | ROC-AUC | F1 | Recall | FPR")
    for name in ("v5b", "v5d", "v5f", "v5g"):
        print(name.upper() + " | " + " | ".join(f"{metrics[name][k]:.6f}" for k in names))
    print(f"Calibrated delta V5g - V5f: {result['delta_v5g_minus_v5f']}")
    print(f"Saved: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hda_v5g.json")
    parser.add_argument("--training-seed", type=int, choices=(42, 43, 44))
    parser.add_argument("--stage", choices=("fit", "report"), required=True)
    args = parser.parse_args()
    (fit if args.stage == "fit" else report)(args.config, args.training_seed)


if __name__ == "__main__":
    main()
