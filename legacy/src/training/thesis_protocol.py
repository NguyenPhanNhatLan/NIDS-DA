"""Frozen experiment settings and explicit target data roles."""

import hashlib
import json
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]


def resolve_source_seed(protocol, source_seed=None):
    fixed = int(protocol["source_pretraining_seed"]) if protocol else 42
    if source_seed is not None:
        if protocol and source_seed != fixed:
            raise ValueError(f"Protocol cố định source seed={fixed}, nhận {source_seed}.")
        return source_seed
    return fixed


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_DIR / path


def load_protocol(path):
    raw = resolve_path(path).read_bytes()
    protocol = json.loads(raw)
    if not protocol["architecture_frozen"] or not protocol["code_sha256"]:
        raise ValueError("Architecture/code snapshot chưa được khóa.")
    for name, expected in protocol["code_sha256"].items():
        actual = hashlib.sha256(resolve_path(name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Frozen code đã thay đổi: {name}. Cần protocol version mới.")
    return protocol, hashlib.sha256(raw).hexdigest()


def validate_training(protocol, version, seed, epochs, batch_size, lr,
                      class_batch_size=None, lambda_conditional=None):
    if version not in ("v2", "v4"):
        raise ValueError("Thesis protocol chỉ so sánh v2 và v4.")
    if seed not in protocol["development_seeds"] + protocol["final_seeds"]:
        raise ValueError("Seed không nằm trong frozen protocol.")
    settings = protocol["training"]
    supplied = {"epochs": epochs, "batch_size": batch_size, "learning_rate": lr}
    if version == "v4":
        supplied.update(class_batch_size=class_batch_size,
                        lambda_conditional=lambda_conditional)
    for name, value in supplied.items():
        if value != settings[name]:
            raise ValueError(f"Frozen {name}={settings[name]}, nhận {value}.")


def evaluation_target(protocol, phase):
    if phase == "development":
        return resolve_path(protocol["target_data"]["development"])
    if phase != "final":
        raise ValueError("Thesis phase phải là development hoặc final.")
    target = protocol["target_data"]
    if not target["final_test"] or not target["untouched_holdout_confirmed"]:
        raise ValueError("Final holdout chưa được xác nhận; không chạy final evaluation.")
    if not target["disjointness_verified"]:
        raise ValueError("Chưa xác minh final holdout tách biệt với adaptation/development.")
    return resolve_path(target["final_test"])
