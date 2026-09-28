import argparse
import hashlib
import json
from pathlib import Path

import torch

from evaluation.baseline import compute_metrics
from evaluation.calibration import calibrate_margin, fit_affine_calibrator, select_threshold_by_fpr
from models.hda_v1 import HDAV1Model
from training.hda_v4 import build_pseudo_pools
from training.hda_v5b import ROOT, code_hashes, file_hash, load_models, load_setup
from training.thesis_protocol import evaluation_target, resolve_path
from training.v6_data import make_loader, make_teacher_loader


def calibration_code_hashes():
    files = ("src/evaluation/calibration.py", "src/evaluation/hda_v5b_calibration.py",
             "src/evaluation/baseline.py", "src/training/v6_data.py",
             "src/training/hda_v4.py", "src/training/hda_v5b.py",
             "src/models/hda_v1.py", "src/models/baseline.py")
    return {name: file_hash(ROOT / name) for name in files}


def data_snapshot(path):
    files = sorted(Path(path).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet files in {path}")
    snapshot = {}
    for file in files:
        digest = hashlib.sha256()
        with file.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        snapshot[file.name] = digest.hexdigest()
    return snapshot


def load_frozen_models(reference_path):
    reference = json.loads(reference_path.read_text())
    checkpoint_path = resolve_path(reference["checkpoint"])
    if file_hash(checkpoint_path) != reference["checkpoint_sha256"]:
        raise ValueError("Frozen V5b checkpoint changed")
    if file_hash(resolve_path(reference["config"])) != reference["config_sha256"]:
        raise ValueError("Frozen V5b config changed")
    _, config_hash, protocol, protocol_hash = load_setup(reference["config"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (checkpoint["config_sha256"] != config_hash
            or checkpoint["protocol_sha256"] != protocol_hash
            or checkpoint["code_sha256"] != code_hashes()
            or checkpoint["seed"] != reference["seed"]
            or checkpoint.get("version") != "v5b"
            or checkpoint.get("architecture") != "hda_v1"):
        raise ValueError("Frozen V5b checkpoint provenance mismatch")
    source, teacher, provenance = load_models(protocol, protocol_hash, reference["seed"], torch.device("cpu"))
    for key, value in provenance.items():
        if checkpoint[key] != value:
            raise ValueError(f"Frozen model dependency changed: {key}")
    student = HDAV1Model(provenance["target_dim"], source)
    student.adapter.load_state_dict(checkpoint["target_adapter_state_dict"])
    for model in (source, teacher, student):
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad = False
    return reference, protocol, provenance, source, teacher, student


def collect_margins(model, loader, labeled=False):
    margins = []
    labels = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            if labeled:
                features, batch_labels = batch
                labels.append(batch_labels.cpu())
            else:
                features = batch
            _, logits = model(features)
            margins.append((logits[:, 1] - logits[:, 0]).cpu().double())
    if not margins:
        raise ValueError("Empty calibration/evaluation loader")
    return torch.cat(margins), torch.cat(labels) if labeled else None


def payload_hash(payload):
    encoded = json.dumps(payload, sort_keys=True, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def save_frozen(path, payload):
    artifact = {"sha256": payload_hash(payload), "frozen": payload}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(artifact, stream, indent=2, allow_nan=False)
        stream.write("\n")


def fit(reference_path, calibration_path):
    if calibration_path.exists():
        raise FileExistsError(f"Calibration already frozen: {calibration_path}")
    reference, protocol, provenance, source, teacher, student = load_frozen_models(reference_path)
    batch_size = protocol["training"]["batch_size"]
    source_path = resolve_path(reference["source_validation"])
    target_path = resolve_path(protocol["target_data"]["adaptation_train"])
    source_snapshot = data_snapshot(source_path)
    target_snapshot = data_snapshot(target_path)
    source_margins, source_labels = collect_margins(
        source, make_loader(source_path, provenance["source_dim"], batch_size), labeled=True,
    )
    policy = reference["threshold_policy"]
    threshold = select_threshold_by_fpr(source_labels.numpy(), source_margins.numpy(), policy["max_fpr"])
    normal, attack, pseudo = build_pseudo_pools(
        teacher, make_teacher_loader(target_path, provenance["target_dim"], batch_size), torch.device("cpu"),
    )
    normal_margins, _ = collect_margins(student, normal.split(batch_size))
    attack_margins, _ = collect_margins(student, attack.split(batch_size))
    parameters = fit_affine_calibrator(source_margins, source_labels, normal_margins, attack_margins)
    if data_snapshot(source_path) != source_snapshot or data_snapshot(target_path) != target_snapshot:
        raise ValueError("Calibration data changed during fitting")
    payload = {
        "reference": str(reference_path), "reference_sha256": file_hash(reference_path),
        "checkpoint_sha256": reference["checkpoint_sha256"], "model_dependencies": provenance,
        "code_sha256": calibration_code_hashes(),
        "a": parameters["a"], "b": parameters["b"], "anchors": parameters,
        "threshold_policy": policy, "source_margin_threshold": threshold,
        "source_probability_threshold": torch.sigmoid(torch.tensor(threshold, dtype=torch.float64)).item(),
        "raw_target_margin_threshold": (threshold - parameters["b"]) / parameters["a"],
        "source_validation_metrics": compute_metrics(source_labels.numpy(), source_margins.numpy(), threshold),
        "source_validation": str(source_path), "source_validation_files": source_snapshot,
        "target_adaptation_train": str(target_path), "target_adaptation_files": target_snapshot,
        "pseudo_label_metadata": pseudo,
        "target_development": str(evaluation_target(protocol, "development")),
        "target_labels_used_for_fit": False,
    }
    save_frozen(calibration_path, payload)
    print(f"Frozen a={parameters['a']:.8f}, b={parameters['b']:.8f}")
    print(f"Source FPR limit={policy['max_fpr']:.4f}, margin threshold={threshold:.8f}")
    print(f"Saved: {calibration_path}")


def report(calibration_path, output_path):
    if output_path.exists():
        raise FileExistsError(f"Development report already exists: {output_path}")
    artifact = json.loads(calibration_path.read_text())
    frozen = artifact["frozen"]
    if payload_hash(frozen) != artifact["sha256"]:
        raise ValueError("Frozen calibration was modified")
    if frozen["code_sha256"] != calibration_code_hashes():
        raise ValueError("Calibration/report code changed after freeze")
    reference_path = Path(frozen["reference"])
    if file_hash(reference_path) != frozen["reference_sha256"]:
        raise ValueError("Frozen model reference or threshold policy changed")
    _, protocol, provenance, _, _, student = load_frozen_models(reference_path)
    if provenance != frozen["model_dependencies"]:
        raise ValueError("Frozen model dependencies changed")
    if (data_snapshot(frozen["source_validation"]) != frozen["source_validation_files"]
            or data_snapshot(frozen["target_adaptation_train"]) != frozen["target_adaptation_files"]):
        raise ValueError("Calibration data changed after freeze")
    target_path = evaluation_target(protocol, "development")
    if str(target_path) != frozen["target_development"]:
        raise ValueError("Development split changed after freeze")
    margins, labels = collect_margins(
        student, make_loader(target_path, provenance["target_dim"], protocol["training"]["batch_size"]),
        labeled=True,
    )
    calibrated = calibrate_margin(margins, frozen["a"], frozen["b"])
    metrics = compute_metrics(labels.numpy(), calibrated.numpy(), frozen["source_margin_threshold"])
    raw_metrics = compute_metrics(labels.numpy(), margins.numpy(), frozen["raw_target_margin_threshold"])
    result = {
        "experiment": "Frozen V5b asymmetric + affine margin calibration",
        "phase": "development", "calibration_sha256": artifact["sha256"],
        "checkpoint_sha256": frozen["checkpoint_sha256"],
        "target_labels_used_for_fit": False, "target_data": str(target_path),
        "a": frozen["a"], "b": frozen["b"], "threshold_policy": frozen["threshold_policy"],
        "score_space": "calibrated logit margin", "pr_auc_definition": "average precision",
        "raw_margin_roc_auc": raw_metrics["roc_auc"], "raw_margin_pr_auc": raw_metrics["pr_auc"],
        **metrics,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    for name in ("f1", "recall", "precision", "fpr", "roc_auc", "pr_auc"):
        print(f"{name}={metrics[name]:.6f}")
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["fit", "report"], required=True)
    parser.add_argument("--reference", default="configs/hda_v5b_calibration_frozen.json")
    parser.add_argument("--calibration", default="results/hda_v5b/asymmetric/calibration/seed42.json")
    parser.add_argument("--output", default="results/hda_v5b/asymmetric/development/v5b_calibrated_seed42.json")
    args = parser.parse_args()
    calibration_path = resolve_path(args.calibration)
    if args.stage == "fit":
        fit(resolve_path(args.reference), calibration_path)
    else:
        report(calibration_path, resolve_path(args.output))


if __name__ == "__main__":
    main()
