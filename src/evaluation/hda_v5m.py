"""Fresh V5m calibration on UNSW validation + unlabeled V2 hard anchors."""
import argparse
import json

import torch

from evaluation.baseline import compute_metrics
from evaluation.calibration import calibrate_margin, fit_affine_calibrator, select_threshold_by_fpr
from evaluation.hda_v5b_calibration import (
    build_frozen_target_pools, collect_margins, data_snapshot, payload_hash, save_frozen,
)
from evaluation.hda_v5d import validate_affine
from training.hda_v5b import file_hash
from training.hda_v5m import checkpoint_path, load_context, load_student
from training.thesis_protocol import evaluation_target, resolve_path
from training.v6_data import make_loader


def calibration_path(config):
    return resolve_path(config["calibration_dir"]) / f"seed{config['training_seed']}.json"


def dependencies(config, deps):
    return {**deps, "v5m_checkpoint_sha256": file_hash(checkpoint_path(config))}


def fit(config_path):
    config, protocol, deps, source, v2, _, _ = load_context(config_path)
    output = calibration_path(config)
    if output.exists():
        raise FileExistsError(f"Calibration already exists: {output}")
    student = load_student(config, deps, source)
    frozen_deps = dependencies(config, deps)
    bs = protocol["training"]["batch_size"]
    dims = deps["v5d_provenance"]
    source_path = resolve_path(config["source_validation"])
    target_path = resolve_path(protocol["target_data"]["adaptation_train"])
    source_snapshot, target_snapshot = data_snapshot(source_path), data_snapshot(target_path)
    # V5m retains the original source classifier, so source scoring uses source.
    margins, labels = collect_margins(source, make_loader(source_path, dims["source_dim"], bs), labeled=True)
    threshold = select_threshold_by_fpr(labels.numpy(), margins.numpy(), config["calibration_max_fpr"])
    normal, attack, pseudo = build_frozen_target_pools(v2, protocol, dims["target_dim"], bs)
    normal_margins, _ = collect_margins(student, normal.split(bs))
    attack_margins, _ = collect_margins(student, attack.split(bs))
    params = fit_affine_calibrator(margins, labels, normal_margins, attack_margins)
    validate_affine(params)
    if (source_snapshot != data_snapshot(source_path) or target_snapshot != data_snapshot(target_path)
            or load_context(config_path)[2] != deps or dependencies(config, deps) != frozen_deps):
        raise ValueError("Calibration inputs changed during fitting")
    save_frozen(output, {
        "version": "v5m", "dependencies": frozen_deps, "training_seed": config["training_seed"],
        "source_margin_threshold": threshold, "parameters": params,
        "raw_equivalent_calibrated_threshold": (threshold - params["b"]) / params["a"],
        "max_source_fpr": config["calibration_max_fpr"],
        "source_validation": str(source_path), "source_validation_files": source_snapshot,
        "target_adaptation_train": str(target_path), "target_adaptation_files": target_snapshot,
        "target_development": str(evaluation_target(protocol, "development")),
        "pseudo_metadata": pseudo, "target_labels_used_for_fit": False,
        "source_validation_metrics": compute_metrics(labels.numpy(), margins.numpy(), threshold),
        "protocol": "Independent V5m affine fit: UNSW validation + frozen V2 q02/q95-q98 adaptation-train anchors; no development labels",
    })
    print(f"V5m affine: a={params['a']:.8f}, b={params['b']:.8f}; source threshold={threshold:.8f}")
    print(f"Saved: {output}")


def report(config_path):
    config, protocol, deps, source, _, _, _ = load_context(config_path)
    output = resolve_path(config["result_dir"]) / f"v5m_seed{config['training_seed']}.json"
    if output.exists():
        raise FileExistsError(f"Report already exists: {output}")
    student = load_student(config, deps, source)
    artifact = json.loads(calibration_path(config).read_text())
    frozen = artifact["frozen"]
    if (payload_hash(frozen) != artifact["sha256"]
            or frozen["version"] != "v5m" or frozen["training_seed"] != config["training_seed"]
            or frozen["dependencies"] != dependencies(config, deps)
            or frozen["max_source_fpr"] != config["calibration_max_fpr"]
            or frozen["source_validation_files"] != data_snapshot(resolve_path(config["source_validation"]))
            or frozen["target_adaptation_files"] != data_snapshot(resolve_path(protocol["target_data"]["adaptation_train"]))):
        raise ValueError("Frozen V5m calibration/dependencies changed")
    params = frozen["parameters"]
    validate_affine(params)
    target = evaluation_target(protocol, "development")
    if str(target) != frozen["target_development"]:
        raise ValueError("Development split changed")
    snapshot = data_snapshot(target)
    raw, labels = collect_margins(student, make_loader(
        target, deps["v5d_provenance"]["target_dim"], protocol["training"]["batch_size"]), labeled=True)
    calibrated = calibrate_margin(raw, params["a"], params["b"])
    metrics = compute_metrics(labels.numpy(), calibrated.numpy(), frozen["source_margin_threshold"])
    raw_metrics = compute_metrics(labels.numpy(), raw.numpy(), frozen["source_margin_threshold"])
    if (snapshot != data_snapshot(target) or load_context(config_path)[2] != deps
            or frozen["dependencies"] != dependencies(config, deps)
            or artifact != json.loads(calibration_path(config).read_text())):
        raise ValueError("Inputs changed during report")
    result = {"experiment_id": config["experiment_id"], "phase": "development",
              "training_seed": config["training_seed"], "target_data": str(target),
              "development_files": snapshot, "dependencies": frozen["dependencies"],
              "calibration_sha256": artifact["sha256"], "parameters": params,
              "source_margin_threshold": frozen["source_margin_threshold"],
              "raw_equivalent_calibrated_threshold": frozen["raw_equivalent_calibrated_threshold"],
              "max_source_fpr": frozen["max_source_fpr"], "metrics": {"v5m": metrics},
              "raw_operating_metrics": raw_metrics,
              "composition": "existing V5d adapter + original source classifier",
              "retrained": False, "target_labels_used_for_fit": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print("V5M | AP | ROC-AUC | F1 | Recall | FPR")
    print(" | ".join(f"{metrics[k]:.6f}" for k in ("pr_auc", "roc_auc", "f1", "recall", "fpr")))
    print(f"Saved: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hda_v5m.json")
    parser.add_argument("--stage", choices=("fit", "report"), required=True)
    args = parser.parse_args()
    (fit if args.stage == "fit" else report)(args.config)


if __name__ == "__main__":
    main()
