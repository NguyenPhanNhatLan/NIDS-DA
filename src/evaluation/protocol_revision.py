"""Explicit evaluation revisions bound to an unchanged training protocol."""

import hashlib
import json
import re

from training.thesis_protocol import resolve_path


def load_evaluation_revision(protocol_path, revision_path):
    raw = resolve_path(protocol_path).read_bytes()
    protocol_hash = hashlib.sha256(raw).hexdigest()
    protocol = json.loads(raw)
    revision_raw = resolve_path(revision_path).read_bytes()
    revision = json.loads(revision_raw)
    if revision["training_protocol_sha256"] != protocol_hash:
        raise ValueError("Evaluation revision không khớp training protocol gốc.")
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", revision["evaluation_id"]):
        raise ValueError("evaluation_id không hợp lệ.")
    if not protocol["architecture_frozen"] or not protocol["code_sha256"]:
        raise ValueError("Architecture/code snapshot chưa được khóa.")

    # Only this entry point may differ. baseline.py also participates in training.
    overrides = revision["code_sha256"]
    expected_files = {"src/evaluation/hda.py", "src/evaluation/protocol_revision.py"}
    if set(overrides) != expected_files:
        raise ValueError("Evaluation revision chỉ được đổi HDA evaluation entry point.")
    if "src/evaluation/hda.py" not in protocol["code_sha256"]:
        raise ValueError("Protocol thiếu hash HDA evaluation gốc.")
    hashes = dict(protocol["code_sha256"])
    hashes.update(overrides)
    for name, expected in hashes.items():
        actual = hashlib.sha256(resolve_path(name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Frozen code đã thay đổi: {name}. Cần revision phù hợp.")

    metadata = {
        "evaluation_id": revision["evaluation_id"],
        "evaluation_revision_sha256": hashlib.sha256(revision_raw).hexdigest(),
        "evaluation_code_sha256": overrides,
    }
    return protocol, protocol_hash, metadata
