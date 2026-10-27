import json
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REVISION = "canonical"
MANIFEST = ROOT / "data/revisions" / REVISION / "manifest.json"
SUITE_PATH = ROOT / "configs/proposal_suite_v2.json"
SUITE = json.loads(SUITE_PATH.read_text())
DIRECTIONS = tuple(SUITE["directions"])
SEEDS = tuple(SUITE["seeds"])
METHODS = tuple(SUITE["methods"])
METHOD_CONFIGS = {
    "marginal_mmd": ROOT / "configs/proposal_mmd_v2.json",
    "mk_mmd": ROOT / "configs/proposal_mkmmd_v2.json",
    "class_aware_mmd": ROOT / "configs/proposal_class_aware_v2.json",
}


def experiment_matrix():
    return product(DIRECTIONS, SEEDS, METHODS)


def validate_method_configs():
    if set(METHODS) != {"source_only", *METHOD_CONFIGS}:
        raise ValueError("Suite methods do not match the configured trainers")
    for method, path in METHOD_CONFIGS.items():
        if json.loads(path.read_text())["method"] != method:
            raise ValueError(f"Method does not match its config: {path}")
