import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from models.baseline import BaselineMLP
from models.hda_v1 import HDAV1Model
from models.target_encoder import (
    TargetEncoder,
    TargetModel,
)

from training.baseline import make_loader

from evaluation.baseline import (
    collect_scores,
    compute_metrics,
)


PROJECT_DIR = Path(__file__).resolve().parents[2]

FEATURE_DIR = (
    PROJECT_DIR
    / "data"
    / "features"
)

MODEL_DIR = (
    PROJECT_DIR
    / "models"
)


def diagnose_batch_norm(model, loader):
    """So sánh BN running stats và BN batch stats, không cập nhật model."""
    batch_norms = [
        layer for layer in model.encoder.modules()
        if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm)
    ]
    if not batch_norms:
        print("\n--- BN MODE DIAGNOSTIC ---")
        print("Encoder không có BatchNorm.")
        return

    # Sao lưu running_mean, running_var, num_batches_tracked.
    saved_buffers = [
        {name: value.clone() for name, value in layer.named_buffers()}
        for layer in batch_norms
    ]
    device = next(model.parameters()).device
    all_labels = []
    eval_scores = []
    batch_scores = []
    skipped_rows = 0

    try:
        with torch.no_grad():
            for features, labels in loader:
                # BN train mode không nhận batch chỉ có một dòng.
                if len(labels) < 2:
                    skipped_rows += len(labels)
                    continue

                features = features.to(device)
                # Mỗi batch eval dùng đúng running stats từ checkpoint.
                for layer, original in zip(batch_norms, saved_buffers):
                    for name, value in layer.named_buffers():
                        value.copy_(original[name])
                model.eval()
                _, eval_logits = model(features)

                # Dropout vẫn ở eval; chỉ BN dùng thống kê của batch hiện tại.
                for layer in batch_norms:
                    layer.train()
                _, batch_logits = model(features)

                all_labels.append(labels.numpy())
                eval_scores.append(
                    torch.softmax(eval_logits, dim=1)[:, 1].cpu().numpy()
                )
                batch_scores.append(
                    torch.softmax(batch_logits, dim=1)[:, 1].cpu().numpy()
                )
    finally:
        # Chạy BN ở train mode có thể thay đổi buffers dù đã no_grad.
        for layer, original in zip(batch_norms, saved_buffers):
            for name, value in layer.named_buffers():
                value.copy_(original[name])
        model.eval()

    if not all_labels:
        print("Không có batch >= 2 dòng cho BN diagnostic.")
        return

    labels = np.concatenate(all_labels)
    eval_scores = np.concatenate(eval_scores)
    batch_scores = np.concatenate(batch_scores)
    print("\n--- BN MODE DIAGNOSTIC (không dùng cho report) ---")
    print(f"Đã bỏ {skipped_rows} dòng ở batch cuối có kích thước 1.")
    print(f"Eval-mode score mean: {eval_scores.mean():.6f}")
    print(f"Batch-stat score mean: {batch_scores.mean():.6f}")

    if (labels == 1).any():
        print(f"Eval-mode attack mean: {eval_scores[labels == 1].mean():.6f}")
        print(f"Batch-stat attack mean: {batch_scores[labels == 1].mean():.6f}")
    if len(np.unique(labels)) == 2:
        print(f"Batch-stat AP: {average_precision_score(labels, batch_scores):.4f}")
        print(f"Batch-stat ROC-AUC: {roc_auc_score(labels, batch_scores):.4f}")
    else:
        print("AP/ROC-AUC cần cả hai lớp; không tính cho diagnostic này.")




def collect_logits_and_labels(model, loader):
    """Thu raw logits trên test sau training, không thay đổi trọng số."""
    device = next(model.parameters()).device
    model.eval()
    all_labels = []
    all_logits = []

    with torch.no_grad():
        for features, labels in loader:
            _, logits = model(features.to(device))
            all_labels.append(labels.cpu().numpy())
            all_logits.append(logits.cpu().numpy())

    if not all_labels:
        raise ValueError("Test loader rỗng, không thể tính logit diagnostic.")
    labels = np.concatenate(all_labels)
    logits = np.concatenate(all_logits)
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError(f"Cần logits [rows, 2], nhận {logits.shape}.")
    return labels, logits


def diagnose_logits(model, loader):
    labels, logits = collect_logits_and_labels(model, loader)
    normal_logit = logits[:, 0]
    attack_logit = logits[:, 1]
    margin = attack_logit - normal_logit

    print("\n--- LOGIT DIAGNOSTICS (không dùng cho report) ---")
    print(f"Normal logit mean: {normal_logit.mean():.4f}")
    print(f"Attack logit mean: {attack_logit.mean():.4f}")
    print(f"Margin mean: {margin.mean():.4f}")
    print(f"Margin min: {margin.min():.4f}")
    print(f"Margin max: {margin.max():.4f}")
    for value, class_name in ((0, "Normal"), (1, "Attack")):
        class_margin = margin[labels == value]
        if len(class_margin):
            print(f"{class_name}-class margin mean: {class_margin.mean():.4f}")
        else:
            print(f"{class_name}-class margin mean: N/A (lớp vắng mặt)")

    if len(np.unique(labels)) == 2:
        print(f"Margin AP: {average_precision_score(labels, margin):.4f}")
        print(f"Margin ROC-AUC: {roc_auc_score(labels, margin):.4f}")
    else:
        print("Margin AP/ROC-AUC: N/A (test cần cả hai lớp)")


def collect_latents(model, loader, max_samples=10_000):
    """Thu tối đa max_samples latent trên CPU, chỉ dùng sau khi train."""
    if max_samples < 1:
        raise ValueError("max_samples phải >= 1.")

    device = next(model.parameters()).device
    model.eval()
    all_z = []
    all_y = []
    collected = 0

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            latent, _ = model(features)
            remaining = max_samples - collected
            latent = latent[:remaining]
            labels = labels[:remaining]
            all_z.append(latent.cpu())
            all_y.append(labels.cpu())
            collected += len(latent)
            if collected >= max_samples:
                break

    if not all_z:
        raise ValueError("Loader rỗng, không thể thu latent.")
    return torch.cat(all_z), torch.cat(all_y)


def diagnose_latents(source_model, source_loader, target_model, target_loader):
    """So sánh latent source/target; nhãn target chỉ dùng ở bước diagnostic."""
    source_z, source_y = collect_latents(source_model, source_loader)
    target_z, target_y = collect_latents(target_model, target_loader)
    if source_z.shape[1] != target_z.shape[1]:
        raise ValueError("Số chiều latent source và target không khớp.")

    print("\n--- LATENT STATISTICS (diagnostic only) ---")
    for name, latent in (("Source", source_z), ("Target", target_z)):
        print(f"{name} samples: {len(latent)}")
        print(f"{name} mean: {latent.mean().item():.6f}")
        print(f"{name} std: {latent.std(unbiased=False).item():.6f}")
        print(f"{name} mean norm: {latent.norm(dim=1).mean().item():.6f}")
        print(f"{name} zero ratio: {(latent == 0).float().mean().item():.6f}")
        print(
            f"{name} median per-dimension std: "
            f"{latent.std(dim=0, unbiased=False).median().item():.6f}"
        )

    class_centers = {}
    for domain, latent, labels in (
        ("Source", source_z, source_y),
        ("Target", target_z, target_y),
    ):
        for label, class_name in ((0, "Normal"), (1, "Attack")):
            selected = latent[labels == label]
            print(f"{domain} {class_name} samples: {len(selected)}")
            if len(selected):
                class_centers[(domain, class_name)] = selected.mean(dim=0)

    print("\n--- WITHIN-DOMAIN CLASS SEPARATION (Euclidean) ---")
    for domain in ("Source", "Target"):
        normal_center = class_centers.get((domain, "Normal"))
        attack_center = class_centers.get((domain, "Attack"))
        if normal_center is None or attack_center is None:
            print(f"{domain} class separation: N/A (lớp vắng mặt)")
        else:
            separation = torch.linalg.vector_norm(
                normal_center - attack_center
            ).item()
            print(f"{domain} class separation: {separation:.6f}")

    print("\n--- CLASS CENTROID DISTANCES (Euclidean, diagnostic only) ---")
    for target_class in ("Normal", "Attack"):
        for source_class in ("Normal", "Attack"):
            target_center = class_centers.get(("Target", target_class))
            source_center = class_centers.get(("Source", source_class))
            if target_center is None or source_center is None:
                print(f"Target {target_class} -> Source {source_class}: N/A (lớp vắng mặt)")
                continue
            distance = torch.linalg.vector_norm(target_center - source_center).item()
            print(f"Target {target_class} -> Source {source_class}: {distance:.6f}")

    print("Các khoảng cách trên dùng tối đa 10.000 dòng đầu mỗi tập; "
          "đây là mô tả hình học, không chứng minh chất lượng phân loại.")



def collect_hidden(
    source_model,
    target_model,
    source_loader,
    target_loader,
    max_samples=10_000,
):
    """Thu biểu diễn 256D trước shared fc2/bn2, tối đa max_samples/miền."""
    if max_samples < 1:
        raise ValueError("max_samples phải >= 1.")
    source_model.eval()
    target_model.eval()
    source_hidden = []
    target_hidden = []

    with torch.no_grad():
        for model, loader, collected_hidden, encode in (
            (source_model, source_loader, source_hidden, source_model.encode_hidden),
            (target_model, target_loader, target_hidden, target_model.adapter),
        ):
            device = next(model.parameters()).device
            collected = 0
            for features, _ in loader:
                hidden = encode(features.to(device))
                hidden = hidden[:max_samples - collected]
                collected_hidden.append(hidden.cpu())
                collected += len(hidden)
                if collected >= max_samples:
                    break

    if not source_hidden or not target_hidden:
        raise ValueError("Source hoặc target loader rỗng; không thể đo hidden.")
    source_h = torch.cat(source_hidden)
    target_h = torch.cat(target_hidden)
    if source_h.shape[1] != 256 or target_h.shape[1] != 256:
        raise ValueError("HDAv1 yêu cầu hidden 256 chiều trước shared fc2/bn2.")
    return source_h, target_h


def diagnose_hidden(source_model, target_model, source_loader, target_loader):
    # Dùng đúng hàm MMD của bước train; hàm này tự chọn single-RBF bandwidth.
    from training.adaptation import mmd_loss

    source_h, target_h = collect_hidden(
        source_model, target_model, source_loader, target_loader
    )
    print("\n--- HIDDEN 256D STATISTICS (HDAv1 diagnostic only) ---")
    for name, hidden in (("Source", source_h), ("Target", target_h)):
        print(f"{name} hidden samples: {len(hidden)}")
        print(f"{name} hidden mean: {hidden.mean().item():.6f}")
        print(f"{name} hidden std: {hidden.std(unbiased=False).item():.6f}")
        print(f"{name} hidden norm: {hidden.norm(dim=1).mean().item():.6f}")
        print(f"{name} hidden zero ratio: {(hidden == 0).float().mean().item():.6f}")

    # torch.cdist tạo ma trận pairwise; giới hạn mỗi miền ở 1.000 dòng.
    sample_size = min(1_000, len(source_h), len(target_h))
    with torch.no_grad():
        hidden_mmd, hidden_bw = mmd_loss(
            source_h[:sample_size], target_h[:sample_size]
        )
    print(f"Hidden MMD ({sample_size}/miền): {hidden_mmd.item():.6f}")
    print(f"Hidden bandwidth: {hidden_bw.item():.6f}")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--version",
        choices=["v0", "v1", "v2"],
        default="v0",
    )

    args = parser.parse_args()

    seed = args.seed

    # ======================================================
    # Load source checkpoint
    # ======================================================

    source_checkpoint_path = (
        MODEL_DIR
        / "baselines"
        / f"unsw_seed{seed}.pt"
    )

    source_checkpoint = torch.load(
        source_checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    source_dim = int(
        source_checkpoint[
            "input_dim"
        ]
    )

    source_model = BaselineMLP(
        input_dim=source_dim
    )

    source_model.load_state_dict(
        source_checkpoint[
            "model_state_dict"
        ]
    )

    source_model.eval()

    # ======================================================
    # Load HDA target encoder
    # ======================================================

    if args.version in ("v1", "v2"):
        checkpoint_name = f"unsw_to_cicids_mmd_{args.version}_seed{seed}.pt"
    else:
        checkpoint_name = f"unsw_to_cicids_mmd_seed{seed}.pt"
    hda_checkpoint_path = MODEL_DIR / "hda" / checkpoint_name
    hda_checkpoint = torch.load(
        hda_checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    target_dim = int(hda_checkpoint["target_dim"])
    if int(hda_checkpoint.get("source_dim", source_dim)) != source_dim:
        raise ValueError("HDA checkpoint không khớp source checkpoint.")
    if int(hda_checkpoint.get("seed", seed)) != seed:
        raise ValueError("HDA checkpoint không khớp seed.")

    if args.version in ("v1", "v2"):
        expected_method = (
            "hda_shared_semantic_hidden_mmd"
            if args.version == "v2" else "hda_v1_shared_tail_single_rbf_mmd"
        )
        if hda_checkpoint.get("method") != expected_method:
            raise ValueError(f"Checkpoint không đúng HDA{args.version}.")
        target_model = HDAV1Model(target_dim, source_model)
        target_model.adapter.load_state_dict(
            hda_checkpoint["target_adapter_state_dict"]
        )
    else:
        target_encoder = TargetEncoder(input_dim=target_dim, latent_dim=168)
        target_encoder.load_state_dict(
            hda_checkpoint["target_encoder_state_dict"]
        )
        target_model = TargetModel(target_encoder, source_model.classifier)

    target_model.eval()

    # ======================================================
    # CICIDS TEST
    #
    # labels are used ONLY here.
    # ======================================================

    test_loader = make_loader(
        FEATURE_DIR
        / "cicids_test",
        input_dim=target_dim,
        batch_size=256,
        training=False,
    )

    labels, scores = collect_scores(
        target_model,
        test_loader,
    )
    diagnose_batch_norm(target_model, test_loader)
    diagnose_logits(target_model, test_loader)

    source_loader = make_loader(
        FEATURE_DIR / "unsw_test",
        input_dim=source_dim,
        batch_size=256,
        training=False,
    )
    diagnose_latents(source_model, source_loader, target_model, test_loader)
    if args.version in ("v1", "v2"):
        diagnose_hidden(source_model, target_model, source_loader, test_loader)
    
    print("\n--- SCORE DIAGNOSTICS ---")

    print(
        f"Score min:    {scores.min():.3e}"
    )

    print(
        f"Score max:    {scores.max():.3e}"
    )

    print(
        f"Score mean:   {scores.mean():.3e}"
    )

    print(
        f"Score median: {np.median(scores):.3e}"
    )

    for q in [
        0.01,
        0.05,
        0.25,
        0.50,
        0.75,
        0.95,
        0.99,
    ]:
        print(
            f"q{q:.2f}: "
            f"{np.quantile(scores, q):.3e}"
        )

    # ======================================================
    # Threshold
    #
    # IMPORTANT:
    # Do NOT use CICIDS validation labels.
    #
    # Use threshold selected on source validation.
    # ======================================================

    source_result_path = (
        PROJECT_DIR
        / "results"
        / "baseline"
        / f"unsw_seed{seed}.json"
    )

    with source_result_path.open(
        encoding="utf-8"
    ) as file:

        source_result = json.load(
            file
        )

    threshold = float(
        source_result["threshold"]
    )

    # For your current baseline:
    #
    # threshold ≈ 0.9843

    metrics = compute_metrics(
        labels,
        scores,
        threshold,
    )

    # ======================================================
    # Save
    # ======================================================

    result = {
        "experiment":
            (
                "HDAv2 shared tail + hidden RBF-MMD"
                if args.version == "v2"
                else (
                    "HDAv1 shared tail + marginal RBF-MMD"
                    if args.version == "v1" else "HDA + marginal RBF-MMD"
                )
            ),

        "source_domain":
            "UNSW",

        "target_domain":
            "CICIDS",

        "target_labels_train":
            0,

        "seed":
            seed,

        "version": args.version,
        "alignment_space": hda_checkpoint.get("alignment_space", "latent_168"),

        "threshold_source":
            threshold,

        **metrics,
    }
    
    normal_scores = scores[
    labels == 0
    ]

    attack_scores = scores[
        labels == 1
    ]

    print("\n--- CLASS SCORE DIAGNOSTICS ---")

    print(
        f"Normal mean: "
        f"{normal_scores.mean():.6f}"
    )

    print(
        f"Attack mean: "
        f"{attack_scores.mean():.6f}"
    )

    print(
        f"Normal median: "
        f"{np.median(normal_scores):.6f}"
    )

    print(
        f"Attack median: "
        f"{np.median(attack_scores):.6f}"
    )
    
    for threshold_debug in [
    0.1,
    0.3,
    0.5,
    0.7,
    0.9,
        0.9843,
    ]:
        predictions = (
            scores >= threshold_debug
        )

        positive_rate = (
            predictions.mean()
        )

    print(
        f"threshold="
        f"{threshold_debug:.4f} | "
        f"predicted attack="
        f"{positive_rate:.4f}"
    )

    output_dir = (
        PROJECT_DIR
        / "results"
        / "hda"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_name = (
        f"unsw_to_cicids_mmd_{args.version}_seed{seed}.json"
        if args.version in ("v1", "v2")
        else f"unsw_to_cicids_mmd_seed{seed}.json"
    )
    output_path = output_dir / result_name

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            result,
            file,
            indent=2,
        )

    print(
        f"\nHDA {args.version} UNSW -> CICIDS"
    )

    print(
        f"Threshold (UNSW val): "
        f"{threshold:.4f}"
    )

    print(
        f"AP={metrics['pr_auc']:.4f} | "
        f"ROC-AUC="
        f"{metrics['roc_auc']:.4f} | "
        f"F1={metrics['f1']:.4f} | "
        f"Recall="
        f"{metrics['recall']:.4f} | "
        f"FPR={metrics['fpr']:.4f}"
    )

    print(
        f"Saved: {output_path}"
    )


if __name__ == "__main__":
    main()