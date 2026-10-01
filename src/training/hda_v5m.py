"""Assemble frozen V5m from V5d adapter + original source classifier. NO TRAINING."""
import argparse
import copy
import json

import torch

from evaluation.hda_v5b_calibration import calibration_code_hashes, data_snapshot
from models.hda_v1 import HDAV1Model
from training import hda_v5d
from training.hda_v5b import ROOT, file_hash
from training.thesis_protocol import resolve_path


def code_hashes():
    hashes = {**calibration_code_hashes(), **hda_v5d.code_hashes()}
    for name in ("src/training/hda_v5m.py", "src/evaluation/hda_v5m.py"):
        hashes[name] = file_hash(ROOT / name)
    return hashes


def build_v5m(source, v5d_checkpoint, target_dim):
    # Own copies prevent device/mode changes leaking into the frozen teachers.
    student = HDAV1Model(target_dim, copy.deepcopy(source).cpu())
    student.adapter.load_state_dict(v5d_checkpoint["target_adapter_state_dict"])
    # Intentionally never load v5d_checkpoint['classifier_state_dict'].
    return student.eval().requires_grad_(False)


def load_context(config_path):
    path = resolve_path(config_path)
    config = json.loads(path.read_text())
    if type(config["training_seed"]) is not int or config["training_seed"] != 42:
        raise ValueError("This V5m experiment uses V5d seed42")
    if config["calibration_max_fpr"] != .02:
        raise ValueError("V5m fixes the source FPR budget at 2%")
    if resolve_path(config["source_validation"]).resolve() != (ROOT / "data/features/unsw_val").resolve():
        raise ValueError("Use fixed UNSW validation")
    dc, protocol, provenance, source, v2, v5b = hda_v5d.load_context(
        config["v5d_config"], config["training_seed"])
    checkpoint_path_d = hda_v5d.checkpoint_path(dc)
    checkpoint = torch.load(checkpoint_path_d, map_location="cpu", weights_only=True)
    if (checkpoint.get("version") != "v5d" or checkpoint.get("architecture") != "hda_v5d"
            or checkpoint["provenance"] != provenance
            or checkpoint["code_sha256"] != hda_v5d.code_hashes()
            or checkpoint["training_seed"] != dc["training_seed"]
            or checkpoint["teacher_seed"] != dc["teacher_seed"]
            or checkpoint["classifier_only"] != dc["classifier_only"]):
        raise ValueError("V5d checkpoint provenance mismatch")
    snapshots = {"source": data_snapshot(ROOT / "data/features/unsw_train"),
                 "target": data_snapshot(resolve_path(protocol["target_data"]["adaptation_train"]))}
    if checkpoint["training_data"] != snapshots:
        raise ValueError("V5d training data changed")
    deps = {"config_sha256": file_hash(path), "v5d_provenance": provenance,
            "v5d_checkpoint": str(checkpoint_path_d), "v5d_checkpoint_sha256": file_hash(checkpoint_path_d),
            "code_sha256": code_hashes(), "training_data": snapshots}
    return config, protocol, deps, source, v2, v5b, checkpoint


def checkpoint_path(config):
    return resolve_path(config["checkpoint_dir"]) / f"v5m_seed{config['training_seed']}.pt"


def load_student(config, deps, source):
    checkpoint = torch.load(checkpoint_path(config), map_location="cpu", weights_only=True)
    if (checkpoint.get("version") != "v5m" or checkpoint.get("dependencies") != deps
            or checkpoint.get("classifier_origin") != "original_source"
            or checkpoint.get("retrained") is not False):
        raise ValueError("V5m checkpoint provenance mismatch")
    return build_v5m(source, checkpoint, deps["v5d_provenance"]["target_dim"])


def build(config_path):
    config, _, deps, source, _, _, checkpoint = load_context(config_path)
    output = checkpoint_path(config)
    if output.exists():
        raise FileExistsError(f"V5m already exists: {output}")
    student = build_v5m(source, checkpoint, deps["v5d_provenance"]["target_dim"])
    for name, value in student.classifier.state_dict().items():
        if not torch.equal(value, source.classifier.state_dict()[name]):
            raise ValueError("V5m classifier differs from original source")
    artifact = {"version": "v5m", "architecture": "hda_v1", "dependencies": deps,
                "adapter_origin": "existing_v5d", "classifier_origin": "original_source",
                "retrained": False, "training_seed": config["training_seed"],
                "target_adapter_state_dict": {k: v.detach().cpu() for k, v in student.adapter.state_dict().items()}}
    if load_context(config_path)[2] != deps:
        raise ValueError("Inputs changed during assembly")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        torch.save(artifact, stream)
    print(f"Saved frozen V5m (no training): {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hda_v5m.json")
    args = parser.parse_args()
    build(args.config)


if __name__ == "__main__":
    main()
