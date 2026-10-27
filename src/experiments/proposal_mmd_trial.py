import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import torch

from evaluation.baseline import collect_scores, compute_metrics, select_f1_threshold
from evaluation.proposal_development_lock import require_development_lock
from evaluation.proposal_final_test import validate_development
from experiments.proposal_protocol import DIRECTIONS, METHOD_CONFIGS, ROOT
from features.common_features import COMMON_FEATURES
from models.baseline import BaselineMLP
from project_utils import domains, get_device, sha256
from training.common import set_seed
from training.data_revision import verify_revision
from training.proposal_data import ParquetBatchStream
from training.proposal_mmd import count_classes, load_config, train_mmd


def trial_paths(direction, method, seed, lambda_mmd, tag=None):
    name = tag or f"{method}_lambda{lambda_mmd:g}_{direction}_seed{seed}".replace(".", "p")
    if not name or Path(name).name != name or name in (".", ".."):
        raise ValueError("--tag must be a filename without directories")
    return (ROOT/"models/trials"/f"{name}.pt",
            ROOT/"results/json"/f"trial__{name}.json")


def evaluate(model, folder, seed, threshold):
    loader = ParquetBatchStream(folder, 1024, False, seed, True)
    labels, scores = collect_scores(model, loader)
    return compute_metrics(labels, scores, threshold)


def run(direction, seed=42, method="marginal_mmd", lambda_mmd=None, *, tag=None,
        device_name="auto", threads=4, dry_run=False):
    if direction not in DIRECTIONS or method not in METHOD_CONFIGS:
        raise ValueError("Unknown canonical direction or method")
    if lambda_mmd is None or not math.isfinite(lambda_mmd) or lambda_mmd < 0 or threads < 1:
        raise ValueError("Lambda must be finite/nonnegative and threads positive")
    checkpoint_path, result_path = trial_paths(direction, method, seed, lambda_mmd, tag)
    for path in (checkpoint_path, result_path):
        if path.exists():
            raise FileExistsError(f"Trial output exists; use a different --tag: {path}")

    revision = verify_revision()
    require_development_lock(revision)
    config_path = METHOD_CONFIGS[method]
    config = deepcopy(load_config(config_path))
    config["lambda_mmd"] = float(lambda_mmd)
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    torch.set_num_threads(threads)
    reference = validate_development(direction, "source_only", seed)
    source_result = json.loads(reference["result_path"].read_text())
    source, target = domains(direction)
    base = reference["base"]
    counts = count_classes(base/f"{source}_train", len(COMMON_FEATURES))
    if counts != source_result["source_train_counts"]:
        raise ValueError("Source class counts differ from the frozen baseline")
    device = get_device() if device_name == "auto" else torch.device(device_name)
    print(f"Trial: {method} | {direction} | seed={seed} | lambda={lambda_mmd:g}", flush=True)
    print(f"Device={device}; source counts={counts}; source-only AP={reference['source_val_ap']:.6f}", flush=True)
    print(f"Checkpoint: {checkpoint_path}\nResult: {result_path}", flush=True)
    if dry_run:
        print("Dry run: inputs verified; no training or trial outputs written.")
        return None


    set_seed(seed)
    model = BaselineMLP(len(COMMON_FEATURES)).to(device)
    model.load_state_dict(reference["model"].state_dict())
    batch_size = config["training"]["batch_size"]
    source_train = ParquetBatchStream(base/f"{source}_train", batch_size, True, seed, True, drop_last=True)
    target_train = ParquetBatchStream(base/f"{target}_train", batch_size, True, seed+1000, False, drop_last=True)
    source_val = ParquetBatchStream(base/f"{source}_val", 1024, False, seed, True)
    model, best_epoch, best_ap, history = train_mmd(
        model, counts, source_train, target_train, source_val, config,
        expected_source_ap=reference["source_val_ap"])

    labels, scores = collect_scores(model, source_val)
    threshold = select_f1_threshold(labels, scores)
    source_metrics = compute_metrics(labels, scores, threshold)

    target_metrics = evaluate(model, base/f"{target}_val", seed, threshold)
    baseline_target = source_result["target_development"]
    delta_ap = target_metrics["pr_auc"]-baseline_target["pr_auc"]
    require_development_lock(verify_revision())

    shared = dict(**revision, phase="post_lock_lambda_trial", protocol="proposal_v2_trial",
        direction=direction, seed=seed, method=method, input_dim=len(COMMON_FEATURES),
        features=list(COMMON_FEATURES), lambda_mmd=float(lambda_mmd), alpha_ce=float(config["alpha_ce"]),
        best_epoch=best_epoch, selected_stage="source_pretrained" if best_epoch == 0 else "adaptation",
        best_source_val_ap=best_ap, source_checkpoint=str(reference["checkpoint_path"]),
        source_checkpoint_sha256=sha256(reference["checkpoint_path"]),
        preprocessor_sha256=source_result["preprocessor_sha256"],
        common_feature_config_sha256=source_result["common_feature_config_sha256"],
        prepared_split_sha256=source_result["prepared_split_sha256"],
        config=config, config_sha256=config_hash, config_hash_format="sha256 of sorted compact JSON",
        base_config=str(config_path), base_config_sha256=sha256(config_path),
        target_labels_used_training=False, target_labels_used_checkpoint_selection=False,
        target_labels_used_threshold_selection=False, bn_running_stats_frozen=True,
        selection="source-validation AP including epoch 0; threshold maximizes source-validation Attack F1",
        history=history)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint_path.open("xb") as stream:
        torch.save({**shared, "model_state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()}}, stream)
    result = dict(**shared, checkpoint=str(checkpoint_path), checkpoint_sha256=sha256(checkpoint_path),
        threshold_from_source_val=threshold, source_train_counts=counts, source_val=source_metrics,
        target_development_split=f"{target}_val", target_development=target_metrics,
        source_only_target_ap=baseline_target["pr_auc"], delta_ap_vs_source_only=delta_ap,
        target_validation_labels_used_posthoc=True, target_test_evaluated=False,
        device=str(device), implementation_sha256={str(p.relative_to(ROOT)):sha256(p) for p in (
            Path(__file__).resolve(), ROOT/"src/training/proposal_mmd.py", ROOT/"src/training/adaptation.py",
            ROOT/"src/training/common.py", ROOT/"src/models/baseline.py")})
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Selected stage={shared['selected_stage']}, epoch={best_epoch}, source-val AP={best_ap:.6f}")
    print(f"Target-val AP: {baseline_target['pr_auc']:.6f} → {target_metrics['pr_auc']:.6f}; delta={delta_ap:+.6f}")
    print(f"Saved trial: {result_path}")
    return result


def main():
    parser = argparse.ArgumentParser(description='Run an isolated lambda trial from a frozen source-only checkpoint.\n\nUses the canonical trainer, splits and source-validation selection. Trial outputs\nare separate from the locked experiment matrix; target test is never evaluated.\n')
    parser.add_argument("--direction", choices=DIRECTIONS, default="unsw_to_cicids")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--method", choices=tuple(METHOD_CONFIGS), default="marginal_mmd")
    parser.add_argument("--lambda-mmd", type=float, required=True)
    parser.add_argument("--tag", help="Optional unique trial filename")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true", help="Verify inputs and output paths, without training")
    args = parser.parse_args()
    run(args.direction, args.seed, args.method, args.lambda_mmd, tag=args.tag,
        device_name=args.device, threads=args.threads, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
