"""Evaluate frozen proposal_v2 MLP checkpoints once on the held-out target test split."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from evaluation.baseline import collect_scores, compute_metrics, select_f1_threshold
from features.common_features import COMMON_FEATURES
from models.baseline import BaselineMLP
from training.proposal_data import ParquetBatchStream, split_sha256
from training.proposal_mmd import output_paths, sha256

from training.data_revision import revision_path, verify_revision

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = (
    revision_path("result_root", ROOT / "results/proposal_v2") / "final_target_test"
)
CONFIGS = {
    "marginal_mmd": ROOT / "configs/proposal_mmd_v2.json",
    "mk_mmd": ROOT / "configs/proposal_mkmmd_v2.json",
    "class_aware_mmd": ROOT / "configs/proposal_class_aware_v2.json",
}


def paths(method, direction, seed):
    if method == "source_only":
        return (
            revision_path("model_root", ROOT / "models/proposal_v2")
            / "source_only_target_val"
            / direction
            / f"seed{seed}.pt",
            revision_path("result_root", ROOT / "results/proposal_v2")
            / "source_only_target_val"
            / direction
            / f"seed{seed}.json",
        )
    if method not in CONFIGS:
        raise ValueError(f"Unknown proposal method: {method}")
    return output_paths(direction, seed, CONFIGS[method])


def check_identity(artifact, direction, seed):
    if (
        artifact["direction"] != direction
        or artifact["seed"] != seed
        or artifact["features"] != list(COMMON_FEATURES)
        or artifact.get("input_dim", artifact.get("feature_count"))
        != len(COMMON_FEATURES)
    ):
        raise ValueError(
            "Checkpoint/result direction, seed, or feature schema mismatch"
        )


def validate_development(direction, method, seed=42):
    """Validate identity/provenance/selection and replay source val; never read target test."""
    revision = verify_revision()
    if direction not in {"unsw_to_cicids", "cicids_to_unsw"}:
        raise ValueError(f"Unknown direction: {direction}")
    if method not in {"source_only", *CONFIGS}:
        raise ValueError(f"Unknown method: {method}")
    source, target = (
        ("unsw", "cicids") if direction == "unsw_to_cicids" else ("cicids", "unsw")
    )
    checkpoint_path, result_path = paths(method, direction, seed)
    for path in (checkpoint_path, result_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing frozen experiment artifact: {path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    development = json.loads(result_path.read_text())
    for artifact in (checkpoint, development):
        check_identity(artifact, direction, seed)
        if any(artifact.get(key) != value for key, value in revision.items()):
            raise ValueError('Development checkpoint/result belongs to another frozen revision')
    if development["checkpoint"] != str(checkpoint_path):
        raise ValueError("Development result points to another checkpoint")

    base = revision_path("feature_root", ROOT / "data/features/proposal_v2") / direction
    prepared = {
        "source_train": split_sha256(base / f"{source}_train"),
        "source_val": split_sha256(base / f"{source}_val"),
        "target_train": split_sha256(base / f"{target}_train"),
        "target_val": split_sha256(base / f"{target}_val"),
    }
    common_hash = sha256(ROOT / "configs/common_features_v2.json")
    processor_hash = sha256(
        revision_path("model_root", ROOT / "models/proposal_v2")
        / direction
        / "preprocessor.joblib"
    )
    for artifact in (checkpoint, development):
        if (
            artifact["prepared_split_sha256"] != prepared
            or artifact["common_feature_config_sha256"] != common_hash
            or artifact["preprocessor_sha256"] != processor_hash
        ):
            raise ValueError("Frozen artifact does not match current prepared data")

    if method == "source_only":
        if checkpoint["best_epoch"] != development["best_epoch"] or not np.isclose(
            checkpoint["best_source_val_ap"],
            development["best_source_val_ap"],
            atol=1e-6,
            rtol=0,
        ):
            raise ValueError("Source-only checkpoint selection mismatch")
        threshold = development["threshold_from_source_val"]
        best_ap = checkpoint["best_source_val_ap"]
    else:
        source_checkpoint, _ = paths("source_only", direction, seed)
        if (
            checkpoint["method"] != method
            or development["method"] != method
            or checkpoint["config_sha256"] != sha256(CONFIGS[method])
            or development["config_sha256"] != checkpoint["config_sha256"]
            or checkpoint["source_checkpoint"] != str(source_checkpoint)
            or checkpoint["source_checkpoint_sha256"] != sha256(source_checkpoint)
            or development["source_checkpoint_sha256"]
            != checkpoint["source_checkpoint_sha256"]
            or checkpoint["best_epoch"] != development["best_epoch"]
            or not np.isclose(
                checkpoint["best_source_val_ap"],
                development["best_source_val_ap"],
                atol=1e-6,
                rtol=0,
            )
        ):
            raise ValueError("Adapted artifact provenance or selection mismatch")
        threshold = development["threshold"]
        best_ap = development["best_source_val_ap"]
    if development["target_development_split"] != f"{target}_val":
        raise ValueError("Development result must use target_val")

    model = BaselineMLP(len(COMMON_FEATURES))
    model.load_state_dict(checkpoint["model_state_dict"])
    source_val = ParquetBatchStream(base / f"{source}_val", 1024, False, seed, True)
    source_y, source_scores = collect_scores(model, source_val)
    current_ap = average_precision_score(source_y, source_scores)
    current_threshold = select_f1_threshold(source_y, source_scores)
    if not np.isclose(current_ap, best_ap, atol=1e-3, rtol=0) or not np.isclose(
        current_threshold, threshold, atol=1e-6, rtol=0
    ):
        raise ValueError(
            "Frozen checkpoint no longer reproduces source-val AP/threshold"
        )

    return {'model': model, 'threshold': threshold, 'source_val_ap': float(current_ap),
            'target': target, 'base': base, 'checkpoint_path': checkpoint_path,
            'result_path': result_path, 'revision': revision}


def run(direction, method, seed=42):
    revision = verify_revision()
    if revision:
        from evaluation.proposal_development_lock import require_development_lock
        require_development_lock(revision)
    output = OUTPUT_ROOT / method / direction / f'seed{seed}.json'
    if output.exists():
        raise FileExistsError(f'Final-test result already exists: {output}')
    validated = validate_development(direction, method, seed)
    model, threshold = validated['model'], validated['threshold']
    target, base = validated['target'], validated['base']
    checkpoint_path, result_path = validated['checkpoint_path'], validated['result_path']
    # No target test labels are read until the complete development lock and source replay pass.
    test_path = base / f"{target}_test"
    test_hash = split_sha256(test_path)
    target_y, target_scores = collect_scores(
        model, ParquetBatchStream(test_path, 1024, False, seed, True)
    )
    result = {
        **revision,
        "protocol": "proposal_v2",
        "phase": "final_test",
        "direction": direction,
        "method": method,
        "seed": seed,
        "features": list(COMMON_FEATURES),
        "feature_count": len(COMMON_FEATURES),
        "target_test_split": f"{target}_test",
        "target_test_sha256": test_hash,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "development_result": str(result_path),
        "threshold_from_source_val": threshold,
        "source_val_ap_verified": validated['source_val_ap'],
        "target_test": compute_metrics(target_y, target_scores, threshold),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Saved final target-test result: {output}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--direction", required=True, choices=["unsw_to_cicids", "cicids_to_unsw"]
    )
    parser.add_argument("--method", required=True, choices=["source_only", *CONFIGS])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.direction, args.method, args.seed)


if __name__ == "__main__":
    main()
