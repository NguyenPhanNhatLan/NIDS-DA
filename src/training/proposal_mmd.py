from __future__ import annotations

import argparse
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
from training.proposal_class_aware import (
    audit_pseudo_labels,
    class_aware_mmd_loss,
    confidence_setting,
    class_separation_loss,
    class_aware_batch_stats,
    prepare_class_batch,
)
from training.proposal_data import ParquetBatchStream, split_sha256
from training.proposal_mkmmd import multi_kernel_mmd_loss

from training.data_revision import (
    revision_path,
    file_sha256 as sha256,
    verify_revision,
    require_development_open,
)
from project_utils import domains as direction_domains
from project_utils import get_device

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CONFIG = ROOT / "configs" / "proposal_mmd_v2.json"

COMMON_CONFIG = ROOT / "configs" / "common_features_v2.json"

FEATURE_ROOT = revision_path("feature_root", ROOT / "data/features/proposal_v2")

MODEL_ROOT = revision_path("model_root", ROOT / "models/proposal_v2") / "mmd"

RESULT_ROOT = revision_path("result_root", ROOT / "results/proposal_v2") / "mmd"

SOURCE_ONLY_ROOT = (
    revision_path("result_root", ROOT / "results/proposal_v2")
    / "source_only_target_val"
)
SOURCE_CHECKPOINT_ROOT = (
    revision_path("model_root", ROOT / "models/proposal_v2") / "source_only_target_val"
)


def output_paths(direction, seed, config_path):
    config_path = Path(config_path)
    tag = f"target_val_epoch0_{config_path.stem}"
    model_root, result_root = MODEL_ROOT / tag, RESULT_ROOT / tag
    return (
        model_root / direction / f"seed{seed}.pt",
        result_root / direction / f"seed{seed}.json",
    )


def load_config(path):
    path = Path(path)

    config = json.loads(path.read_text())
    if "selective_alignment" in config:
        raise ValueError("EMA/anchor selective alignment has been removed; use a non-selective config such as proposal_class_aware_ablation_sep010.json")

    if config["feature_count"] != len(COMMON_FEATURES):
        raise ValueError("Config feature_count does not match COMMON_FEATURES")

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
        expected_policy = "skip_batch"
        if config["mmd"].get("missing_class_policy") != expected_policy:
            raise ValueError(
                f"Class-aware policy must be explicitly frozen as {expected_policy}"
            )
        confidence_setting(config)
        kernel = config["mmd"].get("kernel")
        scales = config["mmd"].get("scales")
        if kernel not in {"single_rbf", "multi_rbf"}:
            raise ValueError("Class-aware MMD kernel must be single_rbf or multi_rbf")
        if kernel == "multi_rbf":
            import math
            if not isinstance(scales, list) or len(scales) < 2 or any(
                not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0
                for value in scales
            ):
                raise ValueError("Class-aware multi_rbf requires at least two positive finite scales")
        elif scales is not None:
            raise ValueError("Set multi_rbf to use kernel scales; legacy single_rbf uses no scales")

    if (
        config.get("protocol") != "proposal_v2"
        or config.get("common_feature_config") != "configs/common_features_v2.json"
    ):
        raise ValueError("Only proposal_v2 with common_features_v2.json is supported")

    separation = config.get("class_separation", {})
    if separation:
        import math
        weight = separation.get("weight", 0)
        margin = separation.get("margin", 0.5)
        if config["method"] != "class_aware_mmd":
            raise ValueError("Class separation requires class_aware_mmd")
        if not math.isfinite(weight) or weight < 0 or not math.isfinite(margin) or not 0 < margin <= 4:
            raise ValueError("Separation weight must be finite/nonnegative and squared-distance margin in (0,4]")
    gradient_every = config.get("diagnostics", {}).get("gradient_every_steps", 0)
    if type(gradient_every) is not int or gradient_every < 0:
        raise ValueError("gradient_every_steps must be a nonnegative integer")
    return config


def freeze_bn_stats(model):
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


def weighted_gradient_norm(term, parameters):
    """Global parameter-gradient L2 norm of one weighted loss, without accumulating grads."""
    if not term.requires_grad:
        return 0.0
    gradients = torch.autograd.grad(term, parameters, retain_graph=True, allow_unused=True)
    norms = [gradient.detach().norm() for gradient in gradients if gradient is not None]
    return float(torch.stack(norms).norm()) if norms else 0.0


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
    separation_config = config.get("class_separation", {})
    separation_weight = float(separation_config.get("weight", 0))
    method = config.get("method", "marginal_mmd")
    confidence = confidence_setting(config) if method == "class_aware_mmd" else None
    gradient_every = config.get("diagnostics", {}).get("gradient_every_steps", 0)
    gradient_parameters = tuple(p for p in model.parameters() if p.requires_grad)

    counts = torch.as_tensor(
        counts,
        dtype=torch.float32,
    )

    class_weights = (counts.sum() / (2.0 * counts)).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(
        model.parameters(),
        lr=settings["learning_rate"],
        weight_decay=settings["weight_decay"],
    )

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

    history = [
        {
            "epoch": 0,
            "loss": None,
            "source_ce": None,
            "mmd2": None,
            "bandwidth": None,
            "source_val_ap": best_ap,
            "stage": "source_pretrained",
        }
    ]
    if method == "class_aware_mmd":
        history[0]["pseudo_label_acceptance"] = None
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
        total_separation = 0.0
        accepted_total = 0
        target_total = 0
        accepted_per_class = [0, 0]
        predicted_per_class = [0, 0]
        both_classes_eligible_batches = aligned_batches = 0
        eligibility_batches = {"none": 0, "normal_only": 0, "attack_only": 0, "both": 0}

        gradient_totals = {"source_ce": 0.0, "class_mmd": 0.0, "class_separation": 0.0}
        gradient_samples = 0
        separation_active_batches = 0
        per_class_aligned_batches = [0,0]
        separation_evaluated_batches = 0
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

            if source_x.shape[0] != target_x.shape[0]:
                raise ValueError(
                    "Source and target " "batch sizes differ. " "Use drop_last=True."
                )

            batch_size = len(source_x)

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

            class_batch = None
            if method == "class_aware_mmd":
                class_batch = prepare_class_batch(source_y, target_logits, confidence)
                stats = class_aware_batch_stats(source_y, target_logits, confidence, batch=class_batch)
                separation_evaluated_batches += int(len(stats['eligible_classes'])==2)
                for label in stats['active_classes']:
                    per_class_aligned_batches[label] += int(lambda_mmd>0)
                per_class = stats["accepted_per_class"]
                accepted, seen = sum(per_class), sum(stats["predicted_per_class"])
                accepted_total += accepted
                target_total += seen
                accepted_per_class[0] += per_class[0]
                accepted_per_class[1] += per_class[1]
                predicted_per_class = [
                    a + b
                    for a, b in zip(predicted_per_class, stats["predicted_per_class"])
                ]
                key = {
                    (): "none",
                    (0,): "normal_only",
                    (1,): "attack_only",
                    (0, 1): "both",
                }[tuple(stats["eligible_classes"])]
                eligibility_batches[key] += 1
                both_classes_eligible_batches += int(
                    len(stats["eligible_classes"]) == 2
                )
                aligned_batches += int(
                    lambda_mmd > 0 and bool(stats["active_classes"])
                )

            ce_loss = criterion(
                source_logits,
                source_y,
            )

            if lambda_mmd == 0:
                alignment_loss = source_z.new_zeros(())
                bandwidth = source_z.new_zeros(())
            elif method == "mk_mmd":
                alignment_loss, bandwidth = multi_kernel_mmd_loss(
                    source_z, target_z, config["mmd"]["scales"]
                )
            elif method == "class_aware_mmd":
                alignment_loss, bandwidth = class_aware_mmd_loss(
                    source_z,
                    source_y,
                    target_z,
                    target_logits,
                    confidence,
                    scales=config["mmd"].get("scales"),
                    batch=class_batch,
                )
            else:
                alignment_loss, bandwidth = mmd_loss(source_z, target_z)

            separation_loss = source_z.new_zeros(())
            if separation_weight > 0:
                separation_loss = class_separation_loss(
                    source_z, source_y, target_z, target_logits,
                    confidence,
                    separation_config.get("margin", 0.5),
                    batch=class_batch,
                )
            loss = (alpha_ce * ce_loss + lambda_mmd * alignment_loss
                    + separation_weight * separation_loss)

            if not torch.isfinite(loss):
                raise ValueError("Non-finite MMD " "training loss")

            if gradient_every and steps % gradient_every == 0:
                for name, term in (
                    ("source_ce", alpha_ce * ce_loss),
                    ("class_mmd", lambda_mmd * alignment_loss),
                    ("class_separation", separation_weight * separation_loss),
                ):
                    gradient_totals[name] += weighted_gradient_norm(term, gradient_parameters)
                gradient_samples += 1
            separation_active_batches += int(float(separation_loss.detach()) > 0)
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_loss += float(loss.item())

            total_ce += float(ce_loss.item())

            total_mmd += float(alignment_loss.item())

            total_bandwidth += float(bandwidth.item())
            total_separation += float(separation_loss.item())

            steps += 1

        if steps == 0:
            raise RuntimeError("No MMD training batches")

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
        row["loss_contributions"] = {
            "weighted_source_ce": alpha_ce * total_ce / steps,
            "weighted_class_mmd": lambda_mmd * total_mmd / steps,
            "weighted_class_separation": separation_weight * total_separation / steps,
        }
        if gradient_samples:
            row["gradient_diagnostics"] = {
                "sampled_batches": gradient_samples,
                "every_steps": gradient_every,
                "mean_weighted_parameter_gradient_l2": {
                    name: value / gradient_samples for name, value in gradient_totals.items()
                },
            }
        if method == "class_aware_mmd":
            row["pseudo_label_acceptance"] = {
                "confidence_threshold": confidence,
                "accepted": accepted_total,
                "seen": target_total,
                "rate": accepted_total / target_total,
                "accepted_per_class": accepted_per_class,
                "predicted_per_class": predicted_per_class,
                "acceptance_rate_per_predicted_class": [
                    a / n if n else None
                    for a, n in zip(accepted_per_class, predicted_per_class)
                ],
            }
            row["class_aware_alignment"] = {
                "missing_class_policy": config["mmd"]["missing_class_policy"],
                "active_classes": [c for c,n in enumerate(per_class_aligned_batches) if n],
                "eligible_batch_counts": eligibility_batches,
                "both_classes_eligible_batches": both_classes_eligible_batches,
                "aligned_batches": aligned_batches,
                "skipped_batches": steps - aligned_batches,
                "aligned_batches_per_class": per_class_aligned_batches,
            }

        if separation_weight > 0:
            row["class_separation"] = {
                "loss": total_separation / steps,
                "weight": separation_weight,
                "squared_distance_margin": separation_config.get("margin", 0.5),
                "eligible_batches": separation_evaluated_batches,
                "positive_loss_batches": separation_active_batches,
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
        if gradient_samples:
            norms = row["gradient_diagnostics"]["mean_weighted_parameter_gradient_l2"]
            print(f"  Weighted gradient L2 ({gradient_samples} batches): "
                  f"CE={norms['source_ce']:.6g}; MMD={norms['class_mmd']:.6g}; "
                  f"Sep={norms['class_separation']:.6g}")
        if separation_weight > 0:
            print(f"  Class separation: {total_separation / steps:.6f}; evaluated batches={separation_evaluated_batches}/{steps}")
        if method == "class_aware_mmd":
            print(
                f"  Pseudo-label acceptance: {accepted_total}/{target_total} "
                f"({accepted_total / target_total:.2%}); class 0/1={accepted_per_class}"
            )

        improvement = val_ap - best_ap

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


def run(
    direction,
    seed=42,
    config_path=DEFAULT_CONFIG,
):
    config_path = Path(config_path)

    require_development_open()
    revision = verify_revision()
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

    source_only_path = SOURCE_ONLY_ROOT / direction / f"seed{seed}.json"
    source_checkpoint_path = SOURCE_CHECKPOINT_ROOT / direction / f"seed{seed}.pt"

    if not source_only_path.exists():
        raise FileNotFoundError(
            "Run proposal source-only " "baseline first: " f"{source_only_path}"
        )
    if not source_checkpoint_path.exists():
        raise FileNotFoundError(
            f"Run proposal source-only first: {source_checkpoint_path}"
        )

    source_only = json.loads(source_only_path.read_text())

    if (
        source_only["direction"] != direction
        or source_only["seed"] != seed
        or source_only["feature_count"] != input_dim
        or source_only["features"] != list(COMMON_FEATURES)
        or source_only["target_development_split"] != f"{target}_val"
    ):
        raise ValueError("Source-only reference " "does not match MMD run")

    source_checkpoint = torch.load(
        source_checkpoint_path, map_location="cpu", weights_only=True
    )
    if (
        source_checkpoint["direction"] != direction
        or source_checkpoint["seed"] != seed
        or source_checkpoint["input_dim"] != input_dim
        or source_checkpoint["features"] != list(COMMON_FEATURES)
        or source_checkpoint["best_epoch"] != source_only["best_epoch"]
        or source_checkpoint["common_feature_config_sha256"] != sha256(COMMON_CONFIG)
        or source_checkpoint["preprocessor_sha256"]
        != sha256(
            revision_path("model_root", ROOT / "models/proposal_v2")
            / direction
            / "preprocessor.joblib"
        )
        or source_only["common_feature_config_sha256"]
        != source_checkpoint["common_feature_config_sha256"]
        or source_only["preprocessor_sha256"]
        != source_checkpoint["preprocessor_sha256"]
    ):
        raise ValueError("Source-only checkpoint does not match reference")
    current_splits = {
        "source_train": split_sha256(source_train_path),
        "source_val": split_sha256(source_val_path),
        "target_train": split_sha256(target_train_path),
        "target_val": split_sha256(target_development_path),
    }
    if (
        source_checkpoint.get("prepared_split_sha256") != current_splits
        or source_only.get("prepared_split_sha256") != current_splits
    ):
        raise ValueError("Prepared feature splits changed since source-only training")

    set_seed(seed)

    device = get_device()

    print(f"\n=== {config['method']}: {direction} ===")

    print(f"Source={source} | Target={target}")

    print(f"Seed={seed} | Device={device}")

    print(f"Features={input_dim}")

    print(f"lambda_mmd={config['lambda_mmd']}")
    print(f"alpha_ce={config['alpha_ce']}")
    if config["method"] == "class_aware_mmd":
        print(f"Pseudo-label confidence >= {confidence_setting(config)}")
    if config.get("class_separation"):
        print(f"Class separation: {config['class_separation']}")

    counts = count_classes(
        source_train_path,
        input_dim,
    )

    source_loader = ParquetBatchStream(
        source_train_path,
        config["training"]["batch_size"],
        True,
        seed,
        True,
        drop_last=True,
    )

    target_loader = ParquetBatchStream(
        target_train_path,
        config["training"]["batch_size"],
        True,
        seed + 1000,
        False,
        drop_last=True,
    )

    source_val_loader = ParquetBatchStream(source_val_path, 1024, False, seed, True)

    model = BaselineMLP(input_dim=input_dim).to(device)
    model.load_state_dict(source_checkpoint["model_state_dict"])
    print(f"Loaded source checkpoint: {source_checkpoint_path}")

    pseudo_label_audit = None
    if config["method"] == "class_aware_mmd":
        pseudo_label_audit = audit_pseudo_labels(
            model, target_train_path, confidence_setting(config)
        )

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

    threshold_loader = ParquetBatchStream(source_val_path, 1024, False, seed, True)

    val_labels, val_scores = collect_scores(
        model,
        threshold_loader,
    )

    threshold = select_f1_threshold(
        val_labels,
        val_scores,
    )
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

    model_dir = checkpoint_path.parent

    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            **revision,
            "protocol": config["protocol_id"],
            "method": config["method"],
            "direction": direction,
            "source_domain": source,
            "target_domain": target,
            "seed": seed,
            "input_dim": input_dim,
            "features": list(COMMON_FEATURES),
            "lambda_mmd": config["lambda_mmd"],
            **({"mmd_kernel": config["mmd"]["kernel"], "mmd_scales": config["mmd"]["scales"]}
               if config["mmd"].get("scales") else {}),
            "alpha_ce": config["alpha_ce"],
            "pseudo_label_confidence": confidence_setting(config) if config["method"] == "class_aware_mmd" else None,
            **({"class_separation": config["class_separation"]} if "class_separation" in config else {}),
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
            **(
                {"source_only_pseudo_label_audit": pseudo_label_audit}
                if pseudo_label_audit
                else {}
            ),
            "model_state_dict": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
        },
        checkpoint_path,
    )

    result = {
        **revision,
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
        **({"mmd_scales": config["mmd"]["scales"]} if config["mmd"].get("scales") else {}),
        "bandwidth": config["mmd"]["bandwidth"],
        "lambda_mmd": config["lambda_mmd"],
        "alpha_ce": config["alpha_ce"],
        "pseudo_label_confidence": confidence_setting(config) if config["method"] == "class_aware_mmd" else None,
        **({"class_separation": config["class_separation"]} if "class_separation" in config else {}),
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
        **(
            {"source_only_pseudo_label_audit": pseudo_label_audit}
            if pseudo_label_audit
            else {}
        ),
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
