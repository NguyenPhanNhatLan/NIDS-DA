from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn
import torch.optim as optim

from evaluation.baseline import (
    collect_scores,
    compute_metrics,
    evaluate_ap,
    select_f1_threshold,
)
from features.common_features import COMMON_FEATURES
from models.baseline import BaselineMLP
from training.adaptation import mmd_loss
from training.baseline import set_seed
from training.proposal_class_aware import class_aware_mmd_loss
from training.proposal_data import ParquetBatchStream, split_sha256
from training.proposal_mkmmd import multi_kernel_mmd_loss

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CONFIG = ROOT / "configs" / "proposal_mmd_v1.json"

COMMON_CONFIG = ROOT / "configs" / "common_features_v1.json"

FEATURE_ROOT = ROOT / "data" / "features" / "proposal_v1"

MODEL_ROOT = ROOT / "models" / "proposal_v1" / "mmd"

RESULT_ROOT = ROOT / "results" / "proposal_v1" / "mmd"

SOURCE_ONLY_ROOT = ROOT / "results" / "proposal_v1" / "source_only_target_val"
SOURCE_CHECKPOINT_ROOT = ROOT / "models" / "proposal_v1" / "source_only_target_val"


def output_paths(direction, seed, config_path):
    """Keep pretrained adaptations separate from earlier scratch MMD runs."""
    config_path = Path(config_path)
    tag = f"target_val_epoch0_{config_path.stem}"
    model_root, result_root = MODEL_ROOT / tag, RESULT_ROOT / tag
    return (
        model_root / direction / f"seed{seed}.pt",
        result_root / direction / f"seed{seed}.json",
    )





def sha256(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def load_config(path):
    path = Path(path)

    config = json.loads(path.read_text())

    if config["feature_count"] != len(COMMON_FEATURES):
        raise ValueError("Config feature_count does " "not match COMMON_FEATURES")

    if config["lambda_mmd"] < 0:
        raise ValueError("lambda_mmd must be >= 0")

    if config.get("alpha_ce", 0) <= 0:
        raise ValueError("alpha_ce must be positive")

    if config.get("development_split") != "target_val":
        raise ValueError("Proposal MMD protocol requires development_split=target_val")

    if config.get("method") not in {"marginal_mmd", "mk_mmd", "class_aware_mmd"}:
        raise ValueError("Unsupported proposal alignment method")

    if config["method"] == "mk_mmd" and not config["mmd"].get("scales"):
        raise ValueError("MK-MMD requires bandwidth scales")

    if config["method"] == "class_aware_mmd":
        confidence = config["mmd"].get("target_pseudo_label_confidence")
        if confidence is None or not 0 <= confidence <= 1:
            raise ValueError("Class-aware MMD requires pseudo-label confidence in [0, 1]")

    return config


def direction_domains(direction):

    if direction == "unsw_to_cicids":
        return "unsw", "cicids"

    if direction == "cicids_to_unsw":
        return "cicids", "unsw"

    raise ValueError(f"Unknown direction: {direction}")


def get_device():

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def freeze_bn_stats(model):
    """Keep source-pretrained running mean/variance fixed during adaptation."""
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def count_classes(
    path,
    input_dim,
):

    loader = ParquetBatchStream(path, 4096, False, 0, True)

    counts = torch.zeros(
        2,
        dtype=torch.long,
    )

    for _, labels in loader:

        counts += torch.bincount(
            labels,
            minlength=2,
        )

    if (counts == 0).any():
        raise ValueError(f"Both classes required: " f"{counts.tolist()}")

    return counts.tolist()


# ============================================================
# MMD training
# ============================================================


def train_mmd(
    model,
    counts,
    source_loader,
    target_loader,
    source_val_loader,
    config,
    expected_source_ap=None,
):

    device = next(model.parameters()).device

    settings = config["training"]

    lambda_mmd = float(config["lambda_mmd"])
    alpha_ce = float(config["alpha_ce"])
    method = config.get("method", "marginal_mmd")

    # Same class weighting as source-only
    counts = torch.as_tensor(
        counts,
        dtype=torch.float32,
    )

    class_weights = (counts.sum() / (2.0 * counts)).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Same optimizer as source-only
    optimizer = optim.Adam(
        model.parameters(),
        lr=settings["learning_rate"],
        weight_decay=settings["weight_decay"],
    )

    # The loaded source-only model is a valid candidate. Adaptation must
    # improve source validation AP to replace it.
    model.eval()
    best_ap = evaluate_ap(model, source_val_loader, device)
    if expected_source_ap is not None and abs(best_ap - expected_source_ap) > 1e-3:
        raise ValueError(
            "Source checkpoint no longer matches current source validation data/preprocessing: "
            f"checkpoint AP={expected_source_ap:.6f}, current AP={best_ap:.6f}"
        )
    best_epoch = 0
    best_state = deepcopy(model.state_dict())

    epochs_without_improvement = 0

    history = [{
        "epoch": 0,
        "loss": None,
        "source_ce": None,
        "mmd2": None,
        "bandwidth": None,
        "source_val_ap": best_ap,
        "stage": "source_pretrained",
    }]
    print(f"Epoch 000 | Source pretrained | Val AP={best_ap:.6f}")

    for epoch in range(
        1,
        settings["epochs"] + 1,
    ):

        model.train()
        freeze_bn_stats(model)

        target_iterator = iter(target_loader)

        total_loss = 0.0
        total_ce = 0.0
        total_mmd = 0.0
        total_bandwidth = 0.0

        steps = 0

        # IMPORTANT:
        # source loader defines epoch length.
        # Therefore one epoch corresponds
        # approximately to one source pass,
        # matching source-only training.
        for source_x, source_y in source_loader:

            try:
                target_x = next(target_iterator)

            except StopIteration:

                target_iterator = iter(target_loader)

                target_x = next(target_iterator)

            source_x = source_x.to(device)

            source_y = source_y.to(device)

            target_x = target_x.to(device)

            if source_x.shape[0] != target_x.shape[0]:
                raise ValueError(
                    "Source and target " "batch sizes differ. " "Use drop_last=True."
                )

            batch_size = len(source_x)

            # ------------------------------------------------
            # ONE shared forward pass
            #
            # This avoids source-first / target-second
            # BatchNorm order effects.
            # ------------------------------------------------

            combined_x = torch.cat(
                [
                    source_x,
                    target_x,
                ],
                dim=0,
            )

            (
                combined_z,
                combined_logits,
            ) = model(combined_x)

            source_z = combined_z[:batch_size]

            target_z = combined_z[batch_size:]

            source_logits = combined_logits[:batch_size]
            target_logits = combined_logits[batch_size:]

            # ------------------------------------------------
            # Source supervised loss
            # ------------------------------------------------

            ce_loss = criterion(
                source_logits,
                source_y,
            )

            # ------------------------------------------------
            # Target-unlabeled marginal MMD
            # ------------------------------------------------

            if lambda_mmd == 0:
                alignment_loss = source_z.new_zeros(())
                bandwidth = source_z.new_zeros(())
            elif method == "mk_mmd":
                alignment_loss, bandwidth = multi_kernel_mmd_loss(
                    source_z, target_z, config["mmd"]["scales"]
                )
            elif method == "class_aware_mmd":
                alignment_loss, bandwidth = class_aware_mmd_loss(
                    source_z, source_y, target_z, target_logits,
                    config["mmd"]["target_pseudo_label_confidence"],
                )
            else:
                alignment_loss, bandwidth = mmd_loss(source_z, target_z)

            loss = alpha_ce * ce_loss + lambda_mmd * alignment_loss

            if not torch.isfinite(loss):
                raise ValueError("Non-finite MMD " "training loss")

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_loss += float(loss.item())

            total_ce += float(ce_loss.item())

            total_mmd += float(alignment_loss.item())

            total_bandwidth += float(bandwidth.item())

            steps += 1

        if steps == 0:
            raise RuntimeError("No MMD training batches")

        # ----------------------------------------------------
        # Source validation ONLY
        # ----------------------------------------------------

        val_ap = evaluate_ap(
            model,
            source_val_loader,
            device,
        )

        row = {
            "epoch": epoch,
            "loss": total_loss / steps,
            "source_ce": total_ce / steps,
            "mmd2": total_mmd / steps,
            "bandwidth": total_bandwidth / steps,
            "source_val_ap": val_ap,
        }

        history.append(row)

        print(
            f"Epoch "
            f"{epoch:03d} | "
            f"Loss="
            f"{row['loss']:.6f} | "
            f"CE="
            f"{row['source_ce']:.6f} | "
            f"MMD²="
            f"{row['mmd2']:.6f} | "
            f"Val AP="
            f"{val_ap:.6f}"
        )

        improvement = val_ap - best_ap

        # Same checkpoint rule
        # as source-only baseline
        if val_ap > best_ap:

            best_ap = val_ap
            best_epoch = epoch

            best_state = deepcopy(model.state_dict())

        if improvement > settings["min_delta"]:

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

        if epochs_without_improvement >= settings["patience"]:

            print("Early stopping.")

            break

    if best_state is None:
        raise RuntimeError("No MMD checkpoint selected")

    model.load_state_dict(best_state)

    model.eval()

    return (
        model,
        best_epoch,
        float(best_ap),
        history,
    )


# ============================================================
# Evaluation
# ============================================================


def evaluate_split(
    model,
    path,
    input_dim,
    threshold,
):

    loader = ParquetBatchStream(path, 1024, False, 0, True)

    labels, scores = collect_scores(
        model,
        loader,
    )

    return compute_metrics(
        labels,
        scores,
        threshold,
    )


# ============================================================
# Full run
# ============================================================


def run(
    direction,
    seed=42,
    config_path=DEFAULT_CONFIG,
):

    config_path = Path(config_path)

    config = load_config(config_path)

    source, target = direction_domains(direction)

    checkpoint_path, output_path = output_paths(direction, seed, config_path)
    for path in (checkpoint_path, output_path):
        if path.exists():
            raise FileExistsError(f"Output already exists: {path}")

    input_dim = len(COMMON_FEATURES)

    base = FEATURE_ROOT / direction

    source_train_path = base / f"{source}_train"

    source_val_path = base / f"{source}_val"

    target_train_path = base / f"{target}_train"

    target_development_path = base / f"{target}_val"

    # --------------------------------------------------------
    # Frozen source-only reference
    # --------------------------------------------------------

    source_only_path = SOURCE_ONLY_ROOT / direction / f"seed{seed}.json"
    source_checkpoint_path = SOURCE_CHECKPOINT_ROOT / direction / f"seed{seed}.pt"

    if not source_only_path.exists():
        raise FileNotFoundError(
            "Run proposal source-only " "baseline first: " f"{source_only_path}"
        )
    if not source_checkpoint_path.exists():
        raise FileNotFoundError(f"Run proposal source-only first: {source_checkpoint_path}")

    source_only = json.loads(source_only_path.read_text())

    if (
        source_only["direction"] != direction
        or source_only["seed"] != seed
        or source_only["feature_count"] != input_dim
        or source_only["features"] != list(COMMON_FEATURES)
        or source_only["target_development_split"] != f"{target}_val"
    ):
        raise ValueError("Source-only reference " "does not match MMD run")

    source_checkpoint = torch.load(source_checkpoint_path, map_location="cpu", weights_only=True)
    if (
        source_checkpoint["direction"] != direction
        or source_checkpoint["seed"] != seed
        or source_checkpoint["input_dim"] != input_dim
        or source_checkpoint["features"] != list(COMMON_FEATURES)
        or source_checkpoint["best_epoch"] != source_only["best_epoch"]
        or source_checkpoint["common_feature_config_sha256"] != sha256(COMMON_CONFIG)
        or source_checkpoint["preprocessor_sha256"] != sha256(ROOT / "models/proposal_v1" / direction / "preprocessor.joblib")
        or source_only["common_feature_config_sha256"] != source_checkpoint["common_feature_config_sha256"]
        or source_only["preprocessor_sha256"] != source_checkpoint["preprocessor_sha256"]
    ):
        raise ValueError("Source-only checkpoint does not match reference")
    current_splits = {
        "source_train": split_sha256(source_train_path),
        "source_val": split_sha256(source_val_path),
        "target_train": split_sha256(target_train_path),
        "target_val": split_sha256(target_development_path),
    }
    if (source_checkpoint.get("prepared_split_sha256") != current_splits
            or source_only.get("prepared_split_sha256") != current_splits):
        raise ValueError("Prepared feature splits changed since source-only training")

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    set_seed(seed)

    device = get_device()

    print(f"\n=== {config['method']}: {direction} ===")

    print(f"Source={source} | " f"Target={target}")

    print(f"Seed={seed} | " f"Device={device}")

    print(f"Features={input_dim}")

    print(f"lambda_mmd=" f"{config['lambda_mmd']}")
    print(f"alpha_ce={config['alpha_ce']}")

    # --------------------------------------------------------
    # Source class counts
    # --------------------------------------------------------

    counts = count_classes(
        source_train_path,
        input_dim,
    )

    # --------------------------------------------------------
    # Training loaders
    # --------------------------------------------------------

    source_loader = ParquetBatchStream(
        source_train_path, config["training"]["batch_size"], True, seed, True, drop_last=True
    )

    # This loader reads FEATURES ONLY.
    target_loader = ParquetBatchStream(
        target_train_path, config["training"]["batch_size"], True, seed + 1000,
        False, drop_last=True,
    )

    source_val_loader = ParquetBatchStream(source_val_path, 1024, False, seed, True)

    # --------------------------------------------------------
    # Same architecture as source-only
    # --------------------------------------------------------

    model = BaselineMLP(input_dim=input_dim).to(device)
    model.load_state_dict(source_checkpoint["model_state_dict"])
    print(f"Loaded source checkpoint: {source_checkpoint_path}")

    (
        model,
        best_epoch,
        best_val_ap,
        history,
    ) = train_mmd(
        model,
        counts,
        source_loader,
        target_loader,
        source_val_loader,
        config,
        expected_source_ap=float(source_checkpoint["best_source_val_ap"]),
    )

    # --------------------------------------------------------
    # Threshold:
    # SOURCE VALIDATION ONLY
    # --------------------------------------------------------

    threshold_loader = ParquetBatchStream(source_val_path, 1024, False, seed, True)

    val_labels, val_scores = collect_scores(
        model,
        threshold_loader,
    )

    threshold = select_f1_threshold(
        val_labels,
        val_scores,
    )

    # --------------------------------------------------------
    # Evaluation
    #
    # Target labels are first read HERE,
    # after training/checkpoint/threshold.
    # --------------------------------------------------------

    within_metrics = evaluate_split(
        model,
        source_val_path,
        input_dim,
        threshold,
    )

    cross_metrics = evaluate_split(
        model,
        target_development_path,
        input_dim,
        threshold,
    )

    metric_names = [
        "pr_auc",
        "roc_auc",
        "macro_f1",
        "recall",
        "fpr",
    ]

    baseline_cross = source_only["target_development"]

    delta_vs_source_only = {
        metric: cross_metrics[metric] - baseline_cross[metric]
        for metric in metric_names
    }

    # --------------------------------------------------------
    # Checkpoint
    # --------------------------------------------------------

    model_dir = checkpoint_path.parent

    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "protocol": config["protocol_id"],
            "method": config["method"],
            "direction": direction,
            "source_domain": source,
            "target_domain": target,
            "seed": seed,
            "input_dim": input_dim,
            "features": list(COMMON_FEATURES),
            "lambda_mmd": config["lambda_mmd"],
            "alpha_ce": config["alpha_ce"],
            "best_epoch": best_epoch,
            "selected_stage": "source_pretrained" if best_epoch == 0 else "adaptation",
            "best_source_val_ap": best_val_ap,
            "target_labels_used_training": False,
            "source_checkpoint": str(source_checkpoint_path),
            "source_checkpoint_sha256": sha256(source_checkpoint_path),
            "preprocessor_sha256": source_checkpoint["preprocessor_sha256"],
            "prepared_split_sha256": current_splits,
            "bn_running_stats_frozen": True,
            "config_sha256": sha256(config_path),
            "common_feature_config_sha256": sha256(COMMON_CONFIG),
            "history": history,
            "model_state_dict": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
        },
        checkpoint_path,
    )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    result = {
        "protocol": config["protocol_id"],
        "method": config["method"],
        "phase": "development",
        "target_development_split": f"{target}_val",
        "within_domain_split": f"{source}_val",
        "direction": direction,
        "seed": seed,
        "source_domain": source,
        "target_domain": target,
        "feature_count": input_dim,
        "features": list(COMMON_FEATURES),
        "architecture": "BaselineMLP shared source-target",
        "mmd_layer": "168D latent",
        "kernel": config["mmd"]["kernel"],
        "bandwidth": config["mmd"]["bandwidth"],
        "lambda_mmd": config["lambda_mmd"],
        "alpha_ce": config["alpha_ce"],
        "source_checkpoint": str(source_checkpoint_path),
        "source_checkpoint_sha256": sha256(source_checkpoint_path),
        "preprocessor_sha256": source_checkpoint["preprocessor_sha256"],
        "prepared_split_sha256": current_splits,
        "bn_running_stats_frozen": True,
        "target_labels_used_training": False,
        "target_labels_used_checkpoint_selection": False,
        "target_labels_used_threshold_selection": False,
        "checkpoint_selection": "source validation AP including source-pretrained epoch 0",
        "threshold_selection": "source validation F1",
        "best_epoch": best_epoch,
        "selected_stage": "source_pretrained" if best_epoch == 0 else "adaptation",
        "best_source_val_ap": best_val_ap,
        "threshold": threshold,
        "within_domain": within_metrics,
        "cross_domain": cross_metrics,
        "source_only_reference": {
            metric: baseline_cross[metric] for metric in metric_names
        },
        "delta_vs_source_only": delta_vs_source_only,
        "checkpoint": str(checkpoint_path),
        "config_sha256": sha256(config_path),
        "common_feature_config_sha256": sha256(COMMON_CONFIG),
    }

    output_dir = output_path.parent

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "x",
        encoding="utf-8",
    ) as stream:

        json.dump(
            result,
            stream,
            indent=2,
            allow_nan=False,
        )

        stream.write("\n")

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\nMetric | Source-only " "| MMD | Delta")

    for metric in metric_names:

        print(
            f"{metric:10s} | "
            f"{baseline_cross[metric]:.6f} | "
            f"{cross_metrics[metric]:.6f} | "
            f"{delta_vs_source_only[metric]:+.6f}"
        )

    print(f"\nSaved checkpoint: " f"{checkpoint_path}")

    print(f"Saved result: " f"{output_path}")

    return result


# ============================================================
# CLI
# ============================================================


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--direction",
        required=True,
        choices=[
            "unsw_to_cicids",
            "cicids_to_unsw",
        ],
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
    )

    args = parser.parse_args()

    run(
        args.direction,
        seed=args.seed,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
