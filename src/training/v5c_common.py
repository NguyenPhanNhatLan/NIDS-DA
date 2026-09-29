import json
import math

import torch

from evaluation.hda_v5b_calibration import (
    calibration_code_hashes, data_snapshot, load_frozen_models, payload_hash,
)
from models.hda_v1 import HDAV1Model
from training.hda_v5b import ROOT, file_hash
from training.thesis_protocol import resolve_path


def pipeline_hashes():
    hashes = calibration_code_hashes()
    for name in ("src/training/v5c_common.py", "src/evaluation/v5c_pseudo_audit.py",
                 "src/training/hda_v5c.py", "src/evaluation/hda_v5c.py"):
        hashes[name] = file_hash(ROOT / name) if (ROOT / name).exists() else None
    return hashes


def load_context(config_path):
    config_path = resolve_path(config_path)
    config = json.loads(config_path.read_text())
    if not math.isfinite(config["lambda_pl"]) or config["lambda_pl"] <= 0:
        raise ValueError("lambda_pl must be positive and finite")
    if config["min_samples_per_class"] < 2:
        raise ValueError("At least two accepted samples per class are required")
    if not 0.5 <= config["min_attack_confirmation_rate"] <= 1:
        raise ValueError("Attack confirmation requirement must be between 0.5 and 1")
    if not 0 <= config["calibration_max_fpr"] <= 1:
        raise ValueError("calibration_max_fpr must be in [0, 1]")
    reference_path = resolve_path(config["stage1_reference"])
    reference, protocol, provenance, source, v2, v5b = load_frozen_models(reference_path)
    if reference["seed"] != 42 or provenance["source_seed"] != 42:
        raise ValueError("V5c requires the frozen V5b asymmetric seed-42 teacher")
    if reference["loss_weights"] != {"hidden": 1.0, "normal": 0.05, "attack": 0.02, "rank": 0.10}:
        raise ValueError("Stage 1 must be V5b asymmetric")
    checkpoint = torch.load(resolve_path(reference["checkpoint"]), map_location="cpu", weights_only=True)
    probability = checkpoint["threshold_source"]
    if not 0 < probability < 1:
        raise ValueError("Frozen source probability threshold must be in (0, 1)")
    threshold = math.log(probability / (1 - probability))
    threshold_policy = {"kind": "stage1_source", "v2_margin_threshold": threshold,
                        "v5b_margin_threshold": threshold}
    if config["teacher_calibration"] is not None:
        path = resolve_path(config["teacher_calibration"])
        artifact = json.loads(path.read_text())
        frozen = artifact["frozen"]
        if (payload_hash(frozen) != artifact["sha256"]
                or frozen["checkpoint_sha256"] != reference["checkpoint_sha256"]
                or frozen["model_dependencies"] != provenance
                or not math.isfinite(frozen["a"]) or frozen["a"] <= 0):
            raise ValueError("Teacher calibration does not match frozen Stage 1")
        threshold = (frozen["source_margin_threshold"] - frozen["b"]) / frozen["a"]
        if not math.isfinite(threshold):
            raise ValueError("Teacher margin threshold must be finite")
        threshold_policy.update(kind="frozen_stage1_calibration", v5b_margin_threshold=threshold,
                                calibration_sha256=file_hash(path))
    provenance = {**provenance, "stage1_checkpoint_sha256": reference["checkpoint_sha256"],
                  "stage1_reference_sha256": file_hash(reference_path),
                  "config_sha256": file_hash(config_path), "threshold_policy": threshold_policy}
    return config, protocol, provenance, source, v2, v5b


def audit_paths(config):
    directory = resolve_path(config["audit_dir"])
    return directory / f"seed{config['seed']}.json", directory / f"pools_seed{config['seed']}.pt"


def load_audit(config, protocol, provenance, require_ready=True):
    audit_path, pool_path = audit_paths(config)
    audit = json.loads(audit_path.read_text())
    if audit["provenance"] != provenance or audit["code_sha256"] != pipeline_hashes():
        raise ValueError("Audit config, teacher, threshold or code changed; run a new audit")
    if audit["pool_sha256"] != file_hash(pool_path):
        raise ValueError("Audited pseudo pools changed")
    train_path = resolve_path(protocol["target_data"]["adaptation_train"])
    if audit["target_train_files"] != data_snapshot(train_path):
        raise ValueError("Target adaptation data changed after audit")
    if require_ready and not audit["training_allowed"]:
        raise ValueError("Audit blocked training: " + "; ".join(audit["blocked_reasons"]))
    pools = torch.load(pool_path, map_location="cpu", weights_only=True)
    return audit, pools


def checkpoint_path(config):
    return resolve_path(config["checkpoint_dir"]) / f"v5c_seed{config['seed']}.pt"


def load_student(config, provenance, source):
    path = checkpoint_path(config)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (checkpoint.get("version") != "v5c" or checkpoint["provenance"] != provenance
            or checkpoint["code_sha256"] != pipeline_hashes() or checkpoint["seed"] != config["seed"]):
        raise ValueError("V5c checkpoint provenance/code mismatch")
    audit_path, _ = audit_paths(config)
    if checkpoint["audit_sha256"] != file_hash(audit_path):
        raise ValueError("Training audit changed")
    student = HDAV1Model(provenance["target_dim"], source)
    student.adapter.load_state_dict(checkpoint["target_adapter_state_dict"])
    student.eval()
    for parameter in student.parameters():
        parameter.requires_grad = False
    return student
