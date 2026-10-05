"""Decision-boundary-aware adaptation: weak MMD + Maximum Classifier Discrepancy.

Protocol
--------
- Fixed Common-5 proposal_v2 features and source-fitted preprocessing.
- Start from the existing source-only BaselineMLP checkpoint.
- Target train contributes FEATURES ONLY.
- No target pseudo-labels are used.
- Checkpoint selection uses SOURCE validation AP only.
- Threshold selection uses SOURCE validation F1 only.
- Target validation labels are read only after model/checkpoint/threshold selection.

Each source-target batch uses three MCD stages:
A) Minimize source classification loss w.r.t. G, C1, C2.
B) Freeze G; minimize source CE - lambda_D * target discrepancy w.r.t. C1, C2.
C) Freeze C1/C2; minimize lambda_D * target discrepancy
   + lambda_M * MMD^2(source_z, target_z) w.r.t. G.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from itertools import chain
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn
import torch.optim as optim

from evaluation.baseline import compute_metrics, select_f1_threshold
from features.common_features import COMMON_FEATURES
from models.mcd import MCDModel, load_from_baseline_checkpoint
from training.baseline import set_seed
from training.proposal_data import ParquetBatchStream, split_sha256


ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CONFIG = ROOT / "configs" / "proposal_mcd_v1.json"
COMMON_CONFIG = ROOT / "configs" / "common_features_v2.json"
FEATURE_ROOT = ROOT / "data" / "features" / "proposal_v2"

SOURCE_ONLY_ROOT = ROOT / "results" / "proposal_v2" / "source_only_target_val"
SOURCE_CHECKPOINT_ROOT = ROOT / "models" / "proposal_v2" / "source_only_target_val"

MODEL_ROOT = ROOT / "models" / "proposal_v2" / "mcd"
RESULT_ROOT = ROOT / "results" / "proposal_v2" / "mcd"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def freeze_bn_stats(module):
    for layer in module.modules():
        if isinstance(layer, nn.modules.batchnorm._BatchNorm):
            layer.eval()


def set_requires_grad(module, enabled):
    for parameter in module.parameters():
        parameter.requires_grad = enabled


def load_config(path):
    path = Path(path)
    config = json.loads(path.read_text())

    if config["feature_count"] != len(COMMON_FEATURES):
        raise ValueError("feature_count does not match COMMON_FEATURES")
    if config["method"] != "mmd_mcd":
        raise ValueError("This trainer requires method='mmd_mcd'")
    if float(config["lambda_mmd"]) < 0:
        raise ValueError("lambda_mmd must be >= 0")
    if float(config["lambda_discrepancy"]) <= 0:
        raise ValueError("lambda_discrepancy must be > 0")
    if int(config["generator_steps"]) < 1:
        raise ValueError("generator_steps must be >= 1")
    if float(config["classifier2_noise_std"]) < 0:
        raise ValueError("classifier2_noise_std must be >= 0")

    required_training = {
        "batch_size",
        "epochs",
        "lr_generator",
        "lr_classifier",
        "weight_decay",
        "patience",
        "min_delta",
    }
    missing = required_training - set(config["training"])
    if missing:
        raise ValueError(f"Missing training settings: {sorted(missing)}")

    return config


def count_classes(path):
    loader = ParquetBatchStream(
        path=path,
        batch_size=4096,
        shuffle=False,
        seed=0,
        include_labels=True,
    )
    counts = torch.zeros(2, dtype=torch.long)
    for _, labels in loader:
        counts += torch.bincount(labels, minlength=2)

    if (counts == 0).any():
        raise ValueError(f"Both source classes are required: {counts.tolist()}")
    return counts.tolist()


def mmd_loss(source_z, target_z):
    """Same biased RBF MMD and median rule; reuse kernel distances for bandwidth."""
    size = min(len(source_z), len(target_z))
    source_z, target_z = source_z[:size], target_z[:size]
    d_ss = torch.cdist(source_z, source_z).square()
    d_tt = torch.cdist(target_z, target_z).square()
    d_st = torch.cdist(source_z, target_z).square()
    with torch.no_grad():
        # Combined distance matrix contains both cross-domain blocks.
        distances = torch.cat((d_ss.detach().flatten(), d_tt.detach().flatten(),
                               d_st.detach().flatten(), d_st.detach().flatten()))
        positive = distances[distances > 1e-12]
        bandwidth_squared = (torch.median(positive).clamp_min(1e-6)
                             if positive.numel() else source_z.new_tensor(1.0))
    loss = (torch.exp(-d_ss / (2 * bandwidth_squared)).mean()
            + torch.exp(-d_tt / (2 * bandwidth_squared)).mean()
            - 2 * torch.exp(-d_st / (2 * bandwidth_squared)).mean())
    return loss, bandwidth_squared.sqrt()


def classifier_discrepancy(logits1, logits2):
    p1 = torch.softmax(logits1, dim=1)
    p2 = torch.softmax(logits2, dim=1)
    return torch.abs(p1 - p2).sum(dim=1).mean()


@torch.no_grad()
def collect_scores_mcd(model, loader, device):
    model.eval()

    all_labels = []
    all_scores = []

    for features, labels in loader:
        features = features.to(device)

        z = model.encode(features)
        logits1, logits2 = model.forward_heads(z)

        p1 = torch.softmax(logits1, dim=1)
        p2 = torch.softmax(logits2, dim=1)
        scores = 0.5 * (p1[:, 1] + p2[:, 1])

        all_labels.append(labels.cpu().numpy())
        all_scores.append(scores.cpu().numpy())

    if not all_labels:
        raise ValueError("Evaluation loader is empty.")

    return np.concatenate(all_labels), np.concatenate(all_scores)


def evaluate_ap_mcd(model, loader, device):
    labels, scores = collect_scores_mcd(model, loader, device)
    if len(np.unique(labels)) != 2:
        raise ValueError("Validation must contain both classes.")
    return float(average_precision_score(labels, scores))


def evaluate_split(model, path, threshold, device):
    loader = ParquetBatchStream(
        path=path,
        batch_size=1024,
        shuffle=False,
        seed=0,
        include_labels=True,
    )
    labels, scores = collect_scores_mcd(model, loader, device)
    return compute_metrics(labels, scores, threshold)


def train_mmd_mcd(
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
    lambda_disc = float(config["lambda_discrepancy"])
    generator_steps = int(config["generator_steps"])

    counts_tensor = torch.as_tensor(counts, dtype=torch.float32)
    class_weights = (
        counts_tensor.sum() / (2.0 * counts_tensor)
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer_g = optim.Adam(
        model.encoder.parameters(),
        lr=float(settings["lr_generator"]),
        weight_decay=float(settings["weight_decay"]),
    )
    optimizer_c = optim.Adam(
        chain(
            model.classifier1.parameters(),
            model.classifier2.parameters(),
        ),
        lr=float(settings["lr_classifier"]),
        weight_decay=float(settings["weight_decay"]),
    )

    model.eval()
    best_ap = evaluate_ap_mcd(model, source_val_loader, device)

    if (
        expected_source_ap is not None
        and abs(best_ap - expected_source_ap) > 0.01
    ):
        raise ValueError(
            "MCD initialization is too far from source-only validation AP: "
            f"source-only={expected_source_ap:.6f}, "
            f"mcd_epoch0={best_ap:.6f}"
        )

    best_epoch = 0
    best_state = deepcopy(model.state_dict())
    epochs_without_improvement = 0

    history = [{
        "epoch": 0,
        "stage": "source_pretrained_two_head_init",
        "source_val_ap": float(best_ap),
        "step_a_source_ce": None,
        "step_b_source_ce": None,
        "step_b_target_discrepancy": None,
        "step_c_target_discrepancy": None,
        "step_c_mmd2": None,
        "step_c_bandwidth": None,
    }]

    print(
        f"Epoch 000 | Two-head source init | "
        f"Val AP={best_ap:.6f}"
    )

    for epoch in range(1, int(settings["epochs"]) + 1):
        target_iterator = iter(target_loader)

        epoch_started = perf_counter()
        # Detached device totals avoid transferring every logged scalar to CPU.
        totals = torch.zeros(6, device=device, dtype=torch.float64
                             if device.type != "mps" else torch.float32)
        log_every = int(settings.get("log_every_batches", 250))

        source_steps = 0
        generator_updates = 0

        for source_x, source_y in source_loader:
            try:
                target_x = next(target_iterator)
            except StopIteration:
                target_iterator = iter(target_loader)
                target_x = next(target_iterator)

            source_x = source_x.to(device)
            source_y = source_y.to(device)
            target_x = target_x.to(device)

            if len(source_x) != len(target_x):
                raise ValueError(
                    "Source/target batch sizes differ. Use drop_last=True."
                )

            # ------------------------------------------------------------
            # Step A: source supervision
            # update G, C1 and C2
            # ------------------------------------------------------------
            model.encoder.train()
            model.classifier1.train()
            model.classifier2.train()
            freeze_bn_stats(model.encoder)

            set_requires_grad(model.encoder, True)
            set_requires_grad(model.classifier1, True)
            set_requires_grad(model.classifier2, True)

            source_z = model.encode(source_x)
            source_logits1, source_logits2 = model.forward_heads(source_z)

            source_ce_a = (
                criterion(source_logits1, source_y)
                + criterion(source_logits2, source_y)
            )

            optimizer_g.zero_grad()
            optimizer_c.zero_grad()
            source_ce_a.backward()
            optimizer_g.step()
            optimizer_c.step()

            # ------------------------------------------------------------
            # Step B: maximize target classifier discrepancy
            # freeze G; update C1/C2 while preserving source accuracy
            # ------------------------------------------------------------
            model.encoder.eval()
            model.classifier1.train()
            model.classifier2.train()

            set_requires_grad(model.encoder, False)
            set_requires_grad(model.classifier1, True)
            set_requires_grad(model.classifier2, True)

            with torch.no_grad():
                source_z = model.encode(source_x)
                target_z = model.encode(target_x)

            source_logits1, source_logits2 = model.forward_heads(source_z)
            target_logits1, target_logits2 = model.forward_heads(target_z)

            source_ce_b = (
                criterion(source_logits1, source_y)
                + criterion(source_logits2, source_y)
            )
            target_disc_b = classifier_discrepancy(
                target_logits1,
                target_logits2,
            )

            loss_b = source_ce_b - lambda_disc * target_disc_b

            if not torch.isfinite(loss_b):
                raise ValueError("Non-finite MCD classifier loss.")

            optimizer_c.zero_grad()
            loss_b.backward()
            optimizer_c.step()

            # ------------------------------------------------------------
            # Step C: minimize target discrepancy + weak marginal MMD
            # freeze C1/C2; update G only
            # ------------------------------------------------------------
            set_requires_grad(model.encoder, True)
            set_requires_grad(model.classifier1, False)
            set_requires_grad(model.classifier2, False)

            for _ in range(generator_steps):
                model.encoder.train()
                freeze_bn_stats(model.encoder)
                model.classifier1.eval()
                model.classifier2.eval()

                source_z = model.encode(source_x)
                target_z = model.encode(target_x)

                target_logits1, target_logits2 = model.forward_heads(target_z)
                target_disc_c = classifier_discrepancy(
                    target_logits1,
                    target_logits2,
                )

                if lambda_mmd > 0:
                    mmd2, bandwidth = mmd_loss(source_z, target_z)
                else:
                    mmd2 = source_z.new_zeros(())
                    bandwidth = source_z.new_zeros(())

                loss_c = (
                    lambda_disc * target_disc_c
                    + lambda_mmd * mmd2
                )

                if not torch.isfinite(loss_c):
                    raise ValueError(
                        "Non-finite MMD-MCD generator loss."
                    )

                optimizer_g.zero_grad()
                loss_c.backward()
                optimizer_g.step()

                totals[3:] += torch.stack((target_disc_c.detach(),
                                           mmd2.detach(), bandwidth.detach()))
                generator_updates += 1

            set_requires_grad(model.classifier1, True)
            set_requires_grad(model.classifier2, True)

            totals[:3] += torch.stack((source_ce_a.detach(), source_ce_b.detach(),
                                       target_disc_b.detach()))
            source_steps += 1
            if log_every > 0 and source_steps % log_every == 0:
                # This transfer also waits for device work before timing it.
                mean_ce = float((totals[0] / source_steps).cpu())
                elapsed = perf_counter() - epoch_started
                print(f"Epoch {epoch:03d} | Batch {source_steps} | "
                      f"{source_steps / elapsed:.2f} batches/s | "
                      f"A_CE={mean_ce:.6f}", flush=True)

        if source_steps == 0 or generator_updates == 0:
            raise RuntimeError("No MMD-MCD training batches.")

        set_requires_grad(model.encoder, True)
        set_requires_grad(model.classifier1, True)
        set_requires_grad(model.classifier2, True)

        val_ap = evaluate_ap_mcd(
            model,
            source_val_loader,
            device,
        )

        sum_a_ce, sum_b_ce, sum_b_disc, sum_c_disc, sum_c_mmd, sum_c_bw = totals.cpu().tolist()
        row = {
            "epoch": epoch,
            "epoch_seconds_including_validation": perf_counter() - epoch_started,
            "source_val_ap": float(val_ap),
            "step_a_source_ce": sum_a_ce / source_steps,
            "step_b_source_ce": sum_b_ce / source_steps,
            "step_b_target_discrepancy": sum_b_disc / source_steps,
            "step_c_target_discrepancy": sum_c_disc / generator_updates,
            "step_c_mmd2": sum_c_mmd / generator_updates,
            "step_c_bandwidth": sum_c_bw / generator_updates,
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d} | "
            f"A_CE={row['step_a_source_ce']:.6f} | "
            f"B_Disc={row['step_b_target_discrepancy']:.6f} | "
            f"C_Disc={row['step_c_target_discrepancy']:.6f} | "
            f"MMD²={row['step_c_mmd2']:.6f} | "
            f"Val AP={val_ap:.6f}"
        )

        improvement = val_ap - best_ap

        if val_ap > best_ap:
            best_ap = val_ap
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())

        if improvement > float(settings["min_delta"]):
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= int(settings["patience"]):
            print("Early stopping.")
            break

    model.load_state_dict(best_state)
    model.eval()

    return model, best_epoch, float(best_ap), history


def run(direction, seed=42, config_path=DEFAULT_CONFIG):
    config_path = Path(config_path)
    config = load_config(config_path)

    source, target = direction_domains(direction)
    input_dim = len(COMMON_FEATURES)

    base = FEATURE_ROOT / direction

    source_train_path = base / f"{source}_train"
    source_val_path = base / f"{source}_val"
    target_train_path = base / f"{target}_train"
    target_val_path = base / f"{target}_val"

    source_only_path = (
        SOURCE_ONLY_ROOT / direction / f"seed{seed}.json"
    )
    source_checkpoint_path = (
        SOURCE_CHECKPOINT_ROOT / direction / f"seed{seed}.pt"
    )

    if not source_only_path.is_file():
        raise FileNotFoundError(
            f"Run source-only first: {source_only_path}"
        )
    if not source_checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Run source-only first: {source_checkpoint_path}"
        )

    source_only = json.loads(source_only_path.read_text())
    source_checkpoint = torch.load(
        source_checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    if (
        source_only["direction"] != direction
        or source_only["seed"] != seed
        or source_only["feature_count"] != input_dim
        or source_only["features"] != list(COMMON_FEATURES)
        or source_only["target_development_split"] != f"{target}_val"
    ):
        raise ValueError(
            "Source-only result does not match this MCD run."
        )

    if (
        source_checkpoint["direction"] != direction
        or source_checkpoint["seed"] != seed
        or source_checkpoint["input_dim"] != input_dim
        or source_checkpoint["features"] != list(COMMON_FEATURES)
        or source_checkpoint["best_epoch"] != source_only["best_epoch"]
    ):
        raise ValueError(
            "Source-only checkpoint does not match reference."
        )

    preprocessor_path = (
        ROOT
        / "models"
        / "proposal_v2"
        / direction
        / "preprocessor.joblib"
    )

    if not preprocessor_path.is_file():
        raise FileNotFoundError(
            f"Missing preprocessor: {preprocessor_path}"
        )

    if (
        source_checkpoint["common_feature_config_sha256"]
        != sha256(COMMON_CONFIG)
    ):
        raise ValueError(
            "Common feature config changed since source-only training."
        )

    if (
        source_checkpoint["preprocessor_sha256"]
        != sha256(preprocessor_path)
    ):
        raise ValueError(
            "Preprocessor changed since source-only training."
        )

    current_splits = {
        "source_train": split_sha256(source_train_path),
        "source_val": split_sha256(source_val_path),
        "target_train": split_sha256(target_train_path),
        "target_val": split_sha256(target_val_path),
    }

    if (
        source_checkpoint.get("prepared_split_sha256")
        != current_splits
        or source_only.get("prepared_split_sha256")
        != current_splits
    ):
        raise ValueError(
            "Prepared feature splits changed since source-only training."
        )

    set_seed(seed)
    device = get_device()

    print(f"\n=== MMD-MCD: {direction} ===")
    print(f"Source={source} | Target={target}")
    print(f"Seed={seed} | Device={device}")
    print(f"Features={input_dim}")
    print(f"lambda_mmd={config['lambda_mmd']}")
    print(
        f"lambda_discrepancy={config['lambda_discrepancy']}"
    )
    print(f"generator_steps={config['generator_steps']}")

    counts = count_classes(source_train_path)

    source_loader = ParquetBatchStream(
        path=source_train_path,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        seed=seed,
        include_labels=True,
        drop_last=True,
    )

    target_loader = ParquetBatchStream(
        path=target_train_path,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        seed=seed + 1000,
        include_labels=False,
        drop_last=True,
    )

    source_val_loader = ParquetBatchStream(
        path=source_val_path,
        batch_size=1024,
        shuffle=False,
        seed=seed,
        include_labels=True,
    )

    model = MCDModel(input_dim=input_dim).to(device)

    load_from_baseline_checkpoint(
        model,
        source_checkpoint["model_state_dict"],
        classifier2_noise_std=float(
            config["classifier2_noise_std"]
        ),
    )

    print(
        f"Loaded source checkpoint: {source_checkpoint_path}"
    )

    (
        model,
        best_epoch,
        best_source_val_ap,
        history,
    ) = train_mmd_mcd(
        model=model,
        counts=counts,
        source_loader=source_loader,
        target_loader=target_loader,
        source_val_loader=source_val_loader,
        config=config,
        expected_source_ap=float(
            source_checkpoint["best_source_val_ap"]
        ),
    )

    threshold_loader = ParquetBatchStream(
        path=source_val_path,
        batch_size=1024,
        shuffle=False,
        seed=seed,
        include_labels=True,
    )

    val_labels, val_scores = collect_scores_mcd(
        model,
        threshold_loader,
        device,
    )

    threshold = select_f1_threshold(
        val_labels,
        val_scores,
    )

    within_metrics = compute_metrics(
        val_labels,
        val_scores,
        threshold,
    )

    # Target labels are first read here.
    cross_metrics = evaluate_split(
        model,
        target_val_path,
        threshold,
        device,
    )

    baseline_cross = source_only["target_development"]

    metric_names = [
        "pr_auc",
        "roc_auc",
        "macro_f1",
        "recall",
        "fpr",
    ]

    delta_vs_source_only = {
        metric: cross_metrics[metric] - baseline_cross[metric]
        for metric in metric_names
    }

    tag = config_path.stem

    checkpoint_path = (
        MODEL_ROOT / tag / direction / f"seed{seed}.pt"
    )
    result_path = (
        RESULT_ROOT / tag / direction / f"seed{seed}.json"
    )

    for path in (checkpoint_path, result_path):
        if path.exists():
            raise FileExistsError(
                f"Output already exists: {path}"
            )

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    result_path.parent.mkdir(
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
            "lambda_mmd": float(config["lambda_mmd"]),
            "lambda_discrepancy": float(
                config["lambda_discrepancy"]
            ),
            "generator_steps": int(
                config["generator_steps"]
            ),
            "classifier2_noise_std": float(
                config["classifier2_noise_std"]
            ),
            "best_epoch": best_epoch,
            "selected_stage": (
                "source_pretrained_two_head_init"
                if best_epoch == 0
                else "adaptation"
            ),
            "best_source_val_ap": best_source_val_ap,
            "threshold_from_source_val": threshold,
            "target_labels_used_training": False,
            "target_labels_used_checkpoint_selection": False,
            "target_labels_used_threshold_selection": False,
            "source_checkpoint": str(
                source_checkpoint_path
            ),
            "source_checkpoint_sha256": sha256(
                source_checkpoint_path
            ),
            "preprocessor_sha256": source_checkpoint[
                "preprocessor_sha256"
            ],
            "common_feature_config_sha256": source_checkpoint[
                "common_feature_config_sha256"
            ],
            "prepared_split_sha256": current_splits,
            "history": history,
            "model_state_dict": {
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
            },
        },
        checkpoint_path,
    )

    result = {
        "protocol": config["protocol_id"],
        "method": config["method"],
        "phase": "development",
        "direction": direction,
        "seed": seed,
        "source_domain": source,
        "target_domain": target,
        "feature_count": input_dim,
        "features": list(COMMON_FEATURES),
        "architecture": (
            "Common-5 -> shared 256/168 encoder "
            "-> two 168/32/2 classifiers"
        ),
        "lambda_mmd": float(config["lambda_mmd"]),
        "lambda_discrepancy": float(
            config["lambda_discrepancy"]
        ),
        "generator_steps": int(
            config["generator_steps"]
        ),
        "classifier2_noise_std": float(
            config["classifier2_noise_std"]
        ),
        "checkpoint_selection": (
            "source validation AP including two-head epoch 0"
        ),
        "threshold_selection": "source validation F1",
        "target_labels_used_training": False,
        "target_labels_used_checkpoint_selection": False,
        "target_labels_used_threshold_selection": False,
        "best_epoch": best_epoch,
        "selected_stage": (
            "source_pretrained_two_head_init"
            if best_epoch == 0
            else "adaptation"
        ),
        "best_source_val_ap": best_source_val_ap,
        "threshold": threshold,
        "within_domain": within_metrics,
        "cross_domain": cross_metrics,
        "source_only_reference": {
            metric: baseline_cross[metric]
            for metric in metric_names
        },
        "delta_vs_source_only": delta_vs_source_only,
        "history": history,
        "checkpoint": str(checkpoint_path),
        "source_checkpoint": str(
            source_checkpoint_path
        ),
        "source_checkpoint_sha256": sha256(
            source_checkpoint_path
        ),
        "preprocessor_sha256": source_checkpoint[
            "preprocessor_sha256"
        ],
        "common_feature_config_sha256": source_checkpoint[
            "common_feature_config_sha256"
        ],
        "prepared_split_sha256": current_splits,
    }

    with result_path.open(
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

    print("\nMetric | Source-only | MMD-MCD | Delta")

    for metric in metric_names:
        print(
            f"{metric:10s} | "
            f"{baseline_cross[metric]:.6f} | "
            f"{cross_metrics[metric]:.6f} | "
            f"{delta_vs_source_only[metric]:+.6f}"
        )

    print(f"\nBest epoch: {best_epoch}")
    print(
        f"Best source Val AP: {best_source_val_ap:.6f}"
    )
    print(f"Source threshold: {threshold:.6f}")
    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Saved result: {result_path}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )
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
        default=str(DEFAULT_CONFIG),
    )
    args = parser.parse_args()

    run(
        direction=args.direction,
        seed=args.seed,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
