from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
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
from training.baseline import set_seed
from training.proposal_data import ParquetBatchStream, split_sha256


ROOT = Path(__file__).resolve().parents[2]

COMMON_CONFIG = ROOT / "configs" / "common_features_v2.json"
FEATURE_ROOT = ROOT / "data" / "features" / "proposal_v2"

SOURCE_ONLY_ROOT = ROOT / "results" / "proposal_v2" / "source_only_target_val"
SOURCE_CHECKPOINT_ROOT = ROOT / "models" / "proposal_v2" / "source_only_target_val"

MODEL_ROOT = ROOT / "models" / "proposal_v2" / "class_aware"
RESULT_ROOT = ROOT / "results" / "proposal_v2" / "class_aware"


def direction_domains(direction: str):
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


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_bn_stats(model):
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


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



def estimate_bandwidth_squared(source_z, target_z):
    with torch.no_grad():
        combined = torch.cat(
            [source_z.detach(), target_z.detach()],
            dim=0,
        )

        distance_squared = torch.cdist(combined, combined).square()
        positive = distance_squared[distance_squared > 1e-12]

        if positive.numel() == 0:
            return torch.tensor(
                1.0,
                dtype=combined.dtype,
                device=combined.device,
            )

        return torch.median(positive).clamp_min(1e-6)


def rbf_kernel(x, y, bandwidth_squared):
    distance_squared = torch.cdist(x, y).square()
    return torch.exp(
        -distance_squared / (2.0 * bandwidth_squared)
    )


def mmd_loss_unequal(source_z, target_z):
    if len(source_z) < 2 or len(target_z) < 2:
        raise ValueError("MMD requires at least 2 samples from each side.")

    bandwidth_squared = estimate_bandwidth_squared(
        source_z,
        target_z,
    )

    k_ss = rbf_kernel(source_z, source_z, bandwidth_squared)
    k_tt = rbf_kernel(target_z, target_z, bandwidth_squared)
    k_st = rbf_kernel(source_z, target_z, bandwidth_squared)

    loss = k_ss.mean() + k_tt.mean() - 2.0 * k_st.mean()

    return loss, torch.sqrt(bandwidth_squared)



@torch.no_grad()
def collect_target_margins(
    teacher,
    target_train_path,
    device,
    batch_size=8192,
):

    teacher.eval()

    loader = ParquetBatchStream(
        path=target_train_path,
        batch_size=batch_size,
        shuffle=False,
        seed=0,
        include_labels=False,
    )

    chunks = []

    for target_x in loader:
        target_x = target_x.to(device)

        _, logits = teacher(target_x)

        margin = (
            logits[:, 1] - logits[:, 0]
        ).detach().cpu().numpy().astype(np.float32, copy=False)

        chunks.append(margin)

    if not chunks:
        raise RuntimeError("No target-train rows for pseudo-label threshold estimation.")

    margins = np.concatenate(chunks)

    if not np.isfinite(margins).all():
        raise ValueError("Non-finite teacher margins on target train.")

    return margins


def fit_pseudo_label_thresholds(
    teacher,
    target_train_path,
    device,
    normal_quantile=0.02,
    attack_quantile=0.95,
):
    """Fit fixed target-train ranking thresholds without using target labels."""
    if not 0.0 < normal_quantile < attack_quantile < 1.0:
        raise ValueError(
            "Require 0 < normal_quantile < attack_quantile < 1."
        )

    margins = collect_target_margins(
        teacher,
        target_train_path,
        device,
    )

    q_normal = float(np.quantile(margins, normal_quantile))
    q_attack = float(np.quantile(margins, attack_quantile))

    pseudo_normal = int((margins <= q_normal).sum())
    pseudo_attack = int((margins >= q_attack).sum())

    stats = {
        "target_rows": int(len(margins)),
        "normal_quantile": float(normal_quantile),
        "attack_quantile": float(attack_quantile),
        "q_normal": q_normal,
        "q_attack": q_attack,
        "pseudo_normal_rows": pseudo_normal,
        "pseudo_attack_rows": pseudo_attack,
        "ignored_rows": int(len(margins) - pseudo_normal - pseudo_attack),
        "pseudo_normal_rate": pseudo_normal / len(margins),
        "pseudo_attack_rate": pseudo_attack / len(margins),
        "fixed": True,
        "target_labels_used": False,
    }

    return q_normal, q_attack, stats


@torch.no_grad()
def pseudo_labels_from_teacher(
    teacher,
    target_x,
    q_normal,
    q_attack,
):
    """Assign only extreme-tail target pseudo-labels using fixed teacher margins."""
    teacher.eval()

    _, logits = teacher(target_x)

    margins = logits[:, 1] - logits[:, 0]

    pseudo_y = torch.full(
        (len(target_x),),
        -1,
        dtype=torch.long,
        device=target_x.device,
    )

    normal_mask = margins <= q_normal
    attack_mask = margins >= q_attack

    pseudo_y[normal_mask] = 0
    pseudo_y[attack_mask] = 1

    accepted = normal_mask | attack_mask

    return pseudo_y, accepted


# ============================================================
# Class-aware MMD
# ============================================================

def class_aware_mmd_loss(
    source_z,
    source_y,
    target_z,
    target_pseudo_y,
    target_accepted,
    min_class_samples=4,
):
    """Average normal↔normal and attack↔attack MMD.

    If either class lacks enough source or accepted target samples,
    skip conditional alignment for this batch.
    """
    if min_class_samples < 2:
        raise ValueError("min_class_samples must be >= 2.")

    source_counts = []
    target_counts = []
    class_losses = []
    bandwidths = []

    for label in (0, 1):
        source_class = source_z[source_y == label]
        target_class = target_z[
            target_accepted & (target_pseudo_y == label)
        ]

        source_counts.append(int(len(source_class)))
        target_counts.append(int(len(target_class)))

        if (
            len(source_class) < min_class_samples
            or len(target_class) < min_class_samples
        ):
            zero = source_z.sum() * 0.0

            diagnostics = {
                "used": False,
                "source_counts": source_counts
                + [0] * (2 - len(source_counts)),
                "target_counts": target_counts
                + [0] * (2 - len(target_counts)),
                "normal_mmd2": None,
                "attack_mmd2": None,
            }

            return zero, zero.detach(), diagnostics

        class_loss, bandwidth = mmd_loss_unequal(
            source_class,
            target_class,
        )

        class_losses.append(class_loss)
        bandwidths.append(bandwidth)

    diagnostics = {
        "used": True,
        "source_counts": source_counts,
        "target_counts": target_counts,
        "normal_mmd2": float(class_losses[0].detach().item()),
        "attack_mmd2": float(class_losses[1].detach().item()),
    }

    return (
        torch.stack(class_losses).mean(),
        torch.stack(bandwidths).mean(),
        diagnostics,
    )


# ============================================================
# Training
# ============================================================

def train_class_aware(
    student,
    teacher,
    counts,
    source_loader,
    target_loader,
    source_val_loader,
    q_normal,
    q_attack,
    lambda_mmd=0.001,
    alpha_ce=1.0,
    epochs=50,
    learning_rate=1e-3,
    weight_decay=1e-4,
    patience=7,
    min_delta=1e-4,
    min_class_samples=4,
    expected_source_ap=None,
):
    device = next(student.parameters()).device

    counts_tensor = torch.as_tensor(
        counts,
        dtype=torch.float32,
    )

    class_weights = (
        counts_tensor.sum() / (2.0 * counts_tensor)
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = optim.Adam(
        student.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False

    # Epoch 0 = unchanged source-only checkpoint.
    student.eval()
    best_ap = evaluate_ap(
        student,
        source_val_loader,
        device,
    )

    if (
        expected_source_ap is not None
        and abs(best_ap - expected_source_ap) > 1e-3
    ):
        raise ValueError(
            "Source checkpoint no longer matches current validation data: "
            f"checkpoint AP={expected_source_ap:.6f}, "
            f"current AP={best_ap:.6f}"
        )

    best_epoch = 0
    best_state = deepcopy(student.state_dict())
    epochs_without_improvement = 0

    history = [{
        "epoch": 0,
        "stage": "source_pretrained",
        "loss": None,
        "source_ce": None,
        "conditional_mmd2": None,
        "normal_mmd2": None,
        "attack_mmd2": None,
        "bandwidth": None,
        "source_val_ap": float(best_ap),
        "pseudo_normal_used": None,
        "pseudo_attack_used": None,
        "alignment_batches": None,
        "skipped_batches": None,
    }]

    print(
        f"Epoch 000 | Source pretrained | "
        f"Val AP={best_ap:.6f}"
    )

    for epoch in range(1, epochs + 1):
        student.train()
        freeze_bn_stats(student)
        teacher.eval()

        target_iterator = iter(target_loader)

        total_loss = 0.0
        total_ce = 0.0
        total_mmd = 0.0
        total_bandwidth = 0.0

        normal_mmd_sum = 0.0
        attack_mmd_sum = 0.0

        pseudo_normal_used = 0
        pseudo_attack_used = 0

        alignment_batches = 0
        skipped_batches = 0
        steps = 0

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
                    "Source/target batch sizes differ. "
                    "Use drop_last=True for both loaders."
                )

            batch_size = len(source_x)

            # Student forward: source + target in one pass.
            combined_x = torch.cat(
                [source_x, target_x],
                dim=0,
            )

            combined_z, combined_logits = student(
                combined_x
            )

            source_z = combined_z[:batch_size]
            target_z = combined_z[batch_size:]
            source_logits = combined_logits[:batch_size]

            # Frozen teacher creates target pseudo-labels.
            target_pseudo_y, target_accepted = (
                pseudo_labels_from_teacher(
                    teacher,
                    target_x,
                    q_normal,
                    q_attack,
                )
            )

            ce_loss = criterion(
                source_logits,
                source_y,
            )

            (
                conditional_mmd,
                bandwidth,
                diagnostics,
            ) = class_aware_mmd_loss(
                source_z=source_z,
                source_y=source_y,
                target_z=target_z,
                target_pseudo_y=target_pseudo_y,
                target_accepted=target_accepted,
                min_class_samples=min_class_samples,
            )

            loss = (
                alpha_ce * ce_loss
                + lambda_mmd * conditional_mmd
            )

            if not torch.isfinite(loss):
                raise ValueError(
                    "Non-finite class-aware training loss."
                )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            total_ce += float(ce_loss.item())
            total_mmd += float(conditional_mmd.item())

            pseudo_normal_used += int(
                (
                    target_accepted
                    & (target_pseudo_y == 0)
                ).sum().item()
            )
            pseudo_attack_used += int(
                (
                    target_accepted
                    & (target_pseudo_y == 1)
                ).sum().item()
            )

            if diagnostics["used"]:
                alignment_batches += 1
                total_bandwidth += float(
                    bandwidth.item()
                )
                normal_mmd_sum += diagnostics[
                    "normal_mmd2"
                ]
                attack_mmd_sum += diagnostics[
                    "attack_mmd2"
                ]
            else:
                skipped_batches += 1

            steps += 1

        if steps == 0:
            raise RuntimeError(
                "No class-aware training batches."
            )

        val_ap = evaluate_ap(
            student,
            source_val_loader,
            device,
        )

        row = {
            "epoch": epoch,
            "loss": total_loss / steps,
            "source_ce": total_ce / steps,
            "conditional_mmd2": total_mmd / steps,
            "normal_mmd2": (
                normal_mmd_sum / alignment_batches
                if alignment_batches
                else None
            ),
            "attack_mmd2": (
                attack_mmd_sum / alignment_batches
                if alignment_batches
                else None
            ),
            "bandwidth": (
                total_bandwidth / alignment_batches
                if alignment_batches
                else None
            ),
            "source_val_ap": float(val_ap),
            "pseudo_normal_used": pseudo_normal_used,
            "pseudo_attack_used": pseudo_attack_used,
            "alignment_batches": alignment_batches,
            "skipped_batches": skipped_batches,
        }

        history.append(row)

        print(
            f"Epoch {epoch:03d} | "
            f"Loss={row['loss']:.6f} | "
            f"CE={row['source_ce']:.6f} | "
            f"CondMMD²={row['conditional_mmd2']:.6f} | "
            f"Val AP={val_ap:.6f} | "
            f"Pseudo N/A={pseudo_normal_used}/{pseudo_attack_used} | "
            f"Aligned/Skipped={alignment_batches}/{skipped_batches}"
        )

        improvement = val_ap - best_ap

        if val_ap > best_ap:
            best_ap = val_ap
            best_epoch = epoch
            best_state = deepcopy(
                student.state_dict()
            )

        if improvement > min_delta:
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print("Early stopping.")
            break

    student.load_state_dict(best_state)
    student.eval()

    return (
        student,
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
    threshold,
):
    loader = ParquetBatchStream(
        path=path,
        batch_size=1024,
        shuffle=False,
        seed=0,
        include_labels=True,
    )

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
# Full experiment
# ============================================================

def run(
    direction,
    seed=42,
    lambda_mmd=0.001,
    normal_quantile=0.02,
    attack_quantile=0.95,
    min_class_samples=4,
    epochs=50,
    batch_size=256,
    learning_rate=1e-3,
    weight_decay=1e-4,
    patience=7,
    min_delta=1e-4,
):
    source, target = direction_domains(direction)

    input_dim = len(COMMON_FEATURES)

    base = FEATURE_ROOT / direction

    source_train_path = base / f"{source}_train"
    source_val_path = base / f"{source}_val"
    target_train_path = base / f"{target}_train"
    target_val_path = base / f"{target}_val"

    source_only_path = (
        SOURCE_ONLY_ROOT
        / direction
        / f"seed{seed}.json"
    )

    source_checkpoint_path = (
        SOURCE_CHECKPOINT_ROOT
        / direction
        / f"seed{seed}.pt"
    )

    if not source_only_path.is_file():
        raise FileNotFoundError(
            f"Missing source-only result: {source_only_path}"
        )

    if not source_checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Missing source-only checkpoint: {source_checkpoint_path}"
        )

    source_only = json.loads(
        source_only_path.read_text()
    )

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
    ):
        raise ValueError(
            "Source-only result does not match this run."
        )

    if (
        source_checkpoint["direction"] != direction
        or source_checkpoint["seed"] != seed
        or source_checkpoint["input_dim"] != input_dim
        or source_checkpoint["features"] != list(COMMON_FEATURES)
    ):
        raise ValueError(
            "Source-only checkpoint does not match this run."
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

    print(
        f"\n=== class_aware_mmd_v2: {direction} ==="
    )
    print(
        f"Source={source} | Target={target}"
    )
    print(
        f"Seed={seed} | Device={device}"
    )
    print(
        f"Features={input_dim}"
    )
    print(
        f"lambda_mmd={lambda_mmd}"
    )
    print(
        f"normal_quantile={normal_quantile}"
    )
    print(
        f"attack_quantile={attack_quantile}"
    )
    print(
        f"min_class_samples={min_class_samples}"
    )

    # Teacher and student both start from the SAME frozen source-only checkpoint.
    teacher = BaselineMLP(
        input_dim=input_dim
    ).to(device)

    teacher.load_state_dict(
        source_checkpoint["model_state_dict"]
    )

    teacher.eval()

    for parameter in teacher.parameters():
        parameter.requires_grad = False

    student = BaselineMLP(
        input_dim=input_dim
    ).to(device)

    student.load_state_dict(
        source_checkpoint["model_state_dict"]
    )

    print(
        f"Loaded source checkpoint: {source_checkpoint_path}"
    )

    # Fit global pseudo-label thresholds using TARGET TRAIN FEATURES ONLY.
    q_normal, q_attack, pseudo_stats = (
        fit_pseudo_label_thresholds(
            teacher=teacher,
            target_train_path=target_train_path,
            device=device,
            normal_quantile=normal_quantile,
            attack_quantile=attack_quantile,
        )
    )

    print("\nFrozen-teacher target-train pseudo-label thresholds:")
    print(
        f"  q_normal ({normal_quantile:.3f}) = {q_normal:.6f}"
    )
    print(
        f"  q_attack ({attack_quantile:.3f}) = {q_attack:.6f}"
    )
    print(
        f"  pseudo normal rows = "
        f"{pseudo_stats['pseudo_normal_rows']}"
    )
    print(
        f"  pseudo attack rows = "
        f"{pseudo_stats['pseudo_attack_rows']}"
    )
    print(
        f"  ignored rows = "
        f"{pseudo_stats['ignored_rows']}"
    )

    counts = count_classes(
        source_train_path
    )

    source_loader = ParquetBatchStream(
        path=source_train_path,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        include_labels=True,
        drop_last=True,
    )

    target_loader = ParquetBatchStream(
        path=target_train_path,
        batch_size=batch_size,
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

    (
        student,
        best_epoch,
        best_source_val_ap,
        history,
    ) = train_class_aware(
        student=student,
        teacher=teacher,
        counts=counts,
        source_loader=source_loader,
        target_loader=target_loader,
        source_val_loader=source_val_loader,
        q_normal=q_normal,
        q_attack=q_attack,
        lambda_mmd=lambda_mmd,
        alpha_ce=1.0,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        patience=patience,
        min_delta=min_delta,
        min_class_samples=min_class_samples,
        expected_source_ap=float(
            source_checkpoint["best_source_val_ap"]
        ),
    )

    # Threshold selection uses SOURCE VALIDATION ONLY.
    threshold_loader = ParquetBatchStream(
        path=source_val_path,
        batch_size=1024,
        shuffle=False,
        seed=seed,
        include_labels=True,
    )

    val_labels, val_scores = collect_scores(
        student,
        threshold_loader,
    )

    threshold = select_f1_threshold(
        val_labels,
        val_scores,
    )

    # Target labels are first read here, after training/checkpoint/threshold.
    within_metrics = evaluate_split(
        student,
        source_val_path,
        threshold,
    )

    cross_metrics = evaluate_split(
        student,
        target_val_path,
        threshold,
    )

    metric_names = [
        "pr_auc",
        "roc_auc",
        "macro_f1",
        "recall",
        "fpr",
    ]

    baseline_cross = source_only[
        "target_development"
    ]

    delta_vs_source_only = {
        metric: (
            cross_metrics[metric]
            - baseline_cross[metric]
        )
        for metric in metric_names
    }

    tag = (
        f"q{int(normal_quantile * 100):02d}_"
        f"q{int(attack_quantile * 100):02d}_"
        f"lambda{str(lambda_mmd).replace('.', '')}"
    )

    checkpoint_path = (
        MODEL_ROOT
        / tag
        / direction
        / f"seed{seed}.pt"
    )

    result_path = (
        RESULT_ROOT
        / tag
        / direction
        / f"seed{seed}.json"
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
            "protocol": "proposal_class_aware_v2",
            "method": "class_aware_mmd",
            "direction": direction,
            "source_domain": source,
            "target_domain": target,
            "seed": seed,
            "input_dim": input_dim,
            "features": list(COMMON_FEATURES),
            "lambda_mmd": lambda_mmd,
            "normal_quantile": normal_quantile,
            "attack_quantile": attack_quantile,
            "q_normal": q_normal,
            "q_attack": q_attack,
            "min_class_samples": min_class_samples,
            "best_epoch": best_epoch,
            "selected_stage": (
                "source_pretrained"
                if best_epoch == 0
                else "adaptation"
            ),
            "best_source_val_ap": best_source_val_ap,
            "target_labels_used_training": False,
            "target_labels_used_pseudo_labeling": False,
            "teacher": "frozen source-only checkpoint",
            "pseudo_label_strategy": (
                "global target-train frozen-teacher logit-margin quantiles"
            ),
            "pseudo_label_stats": pseudo_stats,
            "source_checkpoint": str(
                source_checkpoint_path
            ),
            "source_checkpoint_sha256": sha256(
                source_checkpoint_path
            ),
            "preprocessor_sha256": source_checkpoint[
                "preprocessor_sha256"
            ],
            "common_feature_config_sha256": sha256(
                COMMON_CONFIG
            ),
            "prepared_split_sha256": current_splits,
            "bn_running_stats_frozen": True,
            "history": history,
            "model_state_dict": {
                key: value.detach().cpu()
                for key, value
                in student.state_dict().items()
            },
        },
        checkpoint_path,
    )

    result = {
        "protocol": "proposal_class_aware_v2",
        "method": "class_aware_mmd",
        "phase": "development",
        "direction": direction,
        "seed": seed,
        "source_domain": source,
        "target_domain": target,
        "feature_count": input_dim,
        "features": list(COMMON_FEATURES),
        "architecture": "BaselineMLP shared source-target",
        "mmd_layer": "168D latent",
        "kernel": "single_rbf",
        "mmd_estimator": "biased unequal-count",
        "lambda_mmd": lambda_mmd,
        "pseudo_label_strategy": (
            "frozen teacher + global target-train logit-margin quantiles"
        ),
        "normal_quantile": normal_quantile,
        "attack_quantile": attack_quantile,
        "q_normal": q_normal,
        "q_attack": q_attack,
        "min_class_samples": min_class_samples,
        "pseudo_label_stats": pseudo_stats,
        "target_labels_used_training": False,
        "target_labels_used_pseudo_labeling": False,
        "target_labels_used_checkpoint_selection": False,
        "target_labels_used_threshold_selection": False,
        "checkpoint_selection": (
            "source validation AP including source-pretrained epoch 0"
        ),
        "threshold_selection": "source validation F1",
        "best_epoch": best_epoch,
        "selected_stage": (
            "source_pretrained"
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
        "common_feature_config_sha256": sha256(
            COMMON_CONFIG
        ),
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

    print(
        "\nMetric | Source-only | Class-aware MMD | Delta"
    )

    for metric in metric_names:
        print(
            f"{metric:10s} | "
            f"{baseline_cross[metric]:.6f} | "
            f"{cross_metrics[metric]:.6f} | "
            f"{delta_vs_source_only[metric]:+.6f}"
        )

    print(
        f"\nSaved checkpoint: {checkpoint_path}"
    )
    print(
        f"Saved result: {result_path}"
    )

    return result


# ============================================================
# CLI
# ============================================================

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
        "--lambda-mmd",
        type=float,
        default=0.001,
    )

    parser.add_argument(
        "--normal-quantile",
        type=float,
        default=0.02,
    )

    parser.add_argument(
        "--attack-quantile",
        type=float,
        default=0.95,
    )

    parser.add_argument(
        "--min-class-samples",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=7,
    )

    parser.add_argument(
        "--min-delta",
        type=float,
        default=1e-4,
    )

    args = parser.parse_args()

    run(
        direction=args.direction,
        seed=args.seed,
        lambda_mmd=args.lambda_mmd,
        normal_quantile=args.normal_quantile,
        attack_quantile=args.attack_quantile,
        min_class_samples=args.min_class_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        min_delta=args.min_delta,
    )


if __name__ == "__main__":
    main()