"""Fit/report V5j with the same source-threshold and affine policy as V5b."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from evaluation.baseline import compute_metrics
from evaluation.calibration import (
    calibrate_margin,
    fit_affine_calibrator,
    select_threshold_by_fpr,
)
from evaluation.hda_v5b_calibration import (
    build_frozen_target_pools,
    collect_margins,
    data_snapshot,
    payload_hash,
    save_frozen,
)
from training.hda_v5j import (
    ROOT,
    checkpoint_path,
    code_hashes,
    file_hash,
    load_models,
    load_setup,
    load_student,
    load_v5b_reference,
)
from evaluation.hda_v5f import validate_affine
from training.thesis_protocol import evaluation_target, resolve_path
from training.v6_data import make_loader


def calibration_path(config, seed):
    return resolve_path(config["calibration_dir"]) / f"seed{seed}.json"


def result_path(config, seed):
    return resolve_path(config["result_dir"]) / f"v5j_vs_v5b_seed{seed}.json"


def fit(config_path, seed=42):
    config, config_hash, protocol, protocol_hash = load_setup(config_path)
    load_v5b_reference(config, protocol, seed)
    initial_checkpoint_hash = file_hash(checkpoint_path(config, seed))
    initial_code = code_hashes()
    source, teacher, provenance = load_models(
        protocol, protocol_hash, seed, torch.device("cpu")
    )
    checkpoint = torch.load(
        checkpoint_path(config, seed),
        map_location="cpu",
        weights_only=True,
    )
    if (checkpoint.get("version") != "v5j"
            or checkpoint.get("config_sha256") != config_hash
            or checkpoint.get("protocol_sha256") != protocol_hash
            or checkpoint.get("code_sha256") != code_hashes()):
        raise ValueError("V5j checkpoint/config/code mismatch")
    for key, value in provenance.items():
        if checkpoint.get(key) != value:
            raise ValueError(f"V5j frozen dependency mismatch: {key}")

    # Reconstruct trained V5j student.
    _, _, _, _, _, student, _ = load_student(
        config_path, seed, torch.device("cpu")
    )

    output = calibration_path(config, seed)
    if output.exists():
        raise FileExistsError(f"Calibration already exists: {output}")

    source_path = resolve_path(config["source_validation"])
    target_path = resolve_path(protocol["target_data"]["adaptation_train"])
    source_snapshot = data_snapshot(source_path)
    target_snapshot = data_snapshot(target_path)
    batch_size = protocol["training"]["batch_size"]

    # Same source-space threshold policy as frozen V5b.
    source_margins, source_labels = collect_margins(
        source,
        make_loader(
            source_path,
            provenance["source_dim"],
            batch_size,
        ),
        labeled=True,
    )
    threshold = select_threshold_by_fpr(
        source_labels.numpy(),
        source_margins.numpy(),
        config["calibration_max_fpr"],
    )

    # Same V2 hard pseudo anchors as V5b; only the student changed.
    normal, attack, pseudo = build_frozen_target_pools(
        teacher,
        protocol,
        provenance["target_dim"],
        batch_size,
    )
    normal_margins, _ = collect_margins(
        student, normal.split(batch_size)
    )
    attack_margins, _ = collect_margins(
        student, attack.split(batch_size)
    )

    params = fit_affine_calibrator(
        source_margins,
        source_labels,
        normal_margins,
        attack_margins,
    )
    validate_affine(params)

    if (data_snapshot(source_path) != source_snapshot
            or data_snapshot(target_path) != target_snapshot
            or file_hash(checkpoint_path(config, seed)) != initial_checkpoint_hash
            or code_hashes() != initial_code
            or load_setup(config_path)[1] != config_hash):
        raise ValueError("Calibration data changed during fitting")

    payload = {
        "experiment_id": config["experiment_id"],
        "seed": seed,
        "checkpoint_sha256": initial_checkpoint_hash,
        "config_sha256": config_hash,
        "protocol_sha256": protocol_hash,
        "code_sha256": initial_code,
        "model_dependencies": provenance,
        "vat": config["vat"],
        "source_margin_threshold": threshold,
        "a": params["a"],
        "b": params["b"],
        "anchors": params,
        "max_source_fpr": config["calibration_max_fpr"],
        "source_validation_metrics": compute_metrics(
            source_labels.numpy(),
            source_margins.numpy(),
            threshold,
        ),
        "source_validation": str(source_path),
        "source_validation_files": source_snapshot,
        "target_adaptation_train": str(target_path),
        "target_adaptation_files": target_snapshot,
        "target_development": str(
            evaluation_target(protocol, "development")
        ),
        "pseudo_metadata": pseudo,
        "target_labels_used_for_fit": False,
    }

    save_frozen(output, payload)
    print(
        f"V5j affine a={params['a']:.8f}, "
        f"b={params['b']:.8f}, "
        f"source threshold={threshold:.8f}"
    )
    print(f"Saved: {output}")


def report(config_path, seed=42):
    config, config_hash, protocol, protocol_hash = load_setup(config_path)
    output = result_path(config, seed)
    if output.exists():
        raise FileExistsError(f"Report already exists: {output}")
    v5b = load_v5b_reference(config, protocol, seed)
    artifact_path = calibration_path(config, seed)
    artifact = json.loads(artifact_path.read_text())
    frozen = artifact["frozen"]
    validate_affine(frozen)
    if frozen["seed"] != seed or frozen["max_source_fpr"] != config["calibration_max_fpr"]:
        raise ValueError("Calibration seed or FPR policy changed")

    if payload_hash(frozen) != artifact["sha256"]:
        raise ValueError("V5j calibration payload was modified")
    if frozen["config_sha256"] != config_hash:
        raise ValueError("V5j config changed after calibration")
    if frozen["protocol_sha256"] != protocol_hash:
        raise ValueError("V5j protocol changed after calibration")
    if frozen["code_sha256"] != code_hashes():
        raise ValueError("V5j code changed after calibration")
    if frozen["checkpoint_sha256"] != file_hash(
        checkpoint_path(config, seed)
    ):
        raise ValueError("V5j checkpoint changed after calibration")

    _, _, provenance, _, _, student, _ = load_student(
        config_path, seed, torch.device("cpu")
    )
    if provenance != frozen["model_dependencies"]:
        raise ValueError("V5j model dependencies changed")

    if (data_snapshot(frozen["source_validation"])
            != frozen["source_validation_files"]
            or data_snapshot(frozen["target_adaptation_train"])
            != frozen["target_adaptation_files"]):
        raise ValueError("Calibration data changed after fitting")

    target_path = evaluation_target(protocol, "development")
    if str(target_path) != frozen["target_development"]:
        raise ValueError("Development split changed")
    development_snapshot = data_snapshot(target_path)

    margins, labels = collect_margins(
        student,
        make_loader(
            target_path,
            provenance["target_dim"],
            protocol["training"]["batch_size"],
        ),
        labeled=True,
    )
    calibrated = calibrate_margin(
        margins, frozen["a"], frozen["b"]
    )
    metrics = compute_metrics(
        labels.numpy(),
        calibrated.numpy(),
        frozen["source_margin_threshold"],
    )

    # Frozen V5b comparison report from the same development split.
    load_v5b_reference(config, protocol, seed)
    if (development_snapshot != data_snapshot(target_path)
            or file_hash(checkpoint_path(config, seed)) != frozen["checkpoint_sha256"]
            or code_hashes() != frozen["code_sha256"]):
        raise ValueError("Evaluation inputs changed during report")
    v5b_metrics = {
        key: v5b[key]
        for key in ("pr_auc", "roc_auc", "f1", "recall", "fpr")
    }

    names = ("pr_auc", "roc_auc", "f1", "recall", "fpr")
    result = {
        "experiment_id": config["experiment_id"],
        "phase": "development",
        "seed": seed,
        "target_data": str(target_path),
        "development_files": development_snapshot,
        "v5b_reference_report": config["v5b_reference_report"],
        "v5b_reference_sha256": config["v5b_reference_sha256"],
        "target_labels_used_for_training_or_calibration": False,
        "vat": config["vat"],
        "metrics": {
            "v5b": v5b_metrics,
            "v5j": {key: metrics[key] for key in names},
        },
        "delta_v5j_minus_v5b": {
            key: metrics[key] - v5b_metrics[key]
            for key in names
        },
        "operating_point_v5j": metrics,
        "calibration_sha256": artifact["sha256"],
    }

    output = result_path(config, seed)
    if output.exists():
        raise FileExistsError(f"Report already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")

    print("Model | AP | ROC-AUC | F1 | Recall | FPR")
    for name, row in (("V5B", v5b_metrics), ("V5J", metrics)):
        print(
            name + " | " +
            " | ".join(
                f"{row[key]:.6f}" for key in names
            )
        )
    print(f"Delta V5j - V5b: {result['delta_v5j_minus_v5b']}")
    print(f"Saved: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hda_v5j.json")
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), default=42)
    parser.add_argument("--stage", choices=("fit", "report"), required=True)
    args = parser.parse_args()

    if args.stage == "fit":
        fit(args.config, args.seed)
    else:
        report(args.config, args.seed)


if __name__ == "__main__":
    main()
