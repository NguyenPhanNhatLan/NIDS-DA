"""Shared file and experiment helpers; no data or model loading at import time."""

import hashlib
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    path = Path(path)
    if not path.exists():
        from project_utils.result_files import result_file
        path = result_file(path)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def domains(direction):
    if direction == "unsw_to_cicids":
        return "unsw", "cicids"
    if direction == "cicids_to_unsw":
        return "cicids", "unsw"
    raise ValueError(f"Unknown direction: {direction}")


def get_device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
