"""V4: frozen v2 teacher, fixed pseudo-labels, hidden + conditional MMD."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.baseline import BaselineMLP
from models.hda_v1 import HDAV1Model
from training.adaptation import UnlabeledParquetRows, make_unlabeled_loader, mmd_loss
from training.baseline import make_loader, set_seed

PROJECT_DIR = Path(__file__).resolve().parents[2]
FEATURE_DIR = PROJECT_DIR / "data" / "features"
MODEL_DIR = PROJECT_DIR / "models"


def make_teacher_loader(path, input_dim, batch_size):
    # All rows, fixed file/row order, ONLY the features column.
    return DataLoader(
        UnlabeledParquetRows(path, input_dim, training=False),
        batch_size=batch_size, num_workers=0, drop_last=False,
    )


def build_pseudo_pools(teacher, loader, device):
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False

    all_margins = []
    with torch.no_grad():
        for features in loader:
            _, logits = teacher(features.to(device))
            all_margins.append((logits[:, 1] - logits[:, 0]).cpu().numpy())
    if not all_margins:
        raise ValueError("CICIDS train rỗng.")
    margins = np.concatenate(all_margins)
    if not np.isfinite(margins).all():
        raise ValueError("Teacher margins chứa NaN/Inf.")
    q02, q95, q98 = np.quantile(margins, [0.02, 0.95, 0.98])
    normal_parts = []
    attack_parts = []
    offset = 0
    # Same deterministic loader: each margin stays attached to its feature row.
    for features in loader:
        batch_margins = margins[offset:offset + len(features)]
        if len(batch_margins) != len(features):
            raise ValueError("CICIDS train thay đổi giữa hai lượt đọc.")
        normal_parts.append(features[torch.from_numpy(batch_margins <= q02)])
        attack_parts.append(features[torch.from_numpy(
            (batch_margins >= q95) & (batch_margins < q98)
        )])
        offset += len(features)
    if offset != len(margins):
        raise ValueError("CICIDS train thay đổi giữa hai lượt đọc.")
    normal_pool = torch.cat(normal_parts)
    attack_pool = torch.cat(attack_parts)
    if len(normal_pool) < 2 or len(attack_pool) < 2:
        raise ValueError("Pseudo pools quá nhỏ; kiểm tra ties/collapse trong margins.")
    metadata = {
        "q02": float(q02), "q95": float(q95), "q98": float(q98),
        "train_rows": len(margins),
        "pseudo_normal_rows": len(normal_pool),
        "pseudo_attack_rows": len(attack_pool),
        "threshold_domain": "cicids_train",
        "normal_rule": "margin <= q02",
        "attack_rule": "q95 <= margin < q98",
        "fixed": True,
    }
    print(f"Fixed teacher pseudo-labels: {metadata}")
    return normal_pool, attack_pool, metadata


def build_source_pools(source_model, loader, device):
    """Source labels are allowed; cache frozen latent vectors by class on CPU."""
    source_model.eval()
    parts = {0: [], 1: []}
    with torch.no_grad():
        for features, labels in loader:
            hidden = source_model.encode_hidden(features.to(device))
            latent = torch.relu(source_model.bn2(source_model.fc2(hidden))).cpu()
            for label in (0, 1):
                parts[label].append(latent[labels == label])
    if not parts[0]:
        raise ValueError("UNSW train rỗng.")
    pools = {label: torch.cat(parts[label]) for label in (0, 1)}
    if any(len(pool) < 2 for pool in pools.values()):
        raise ValueError("Source conditional pools cần cả Normal và Attack.")
    print(f"Source latent pools: Normal={len(pools[0])}, Attack={len(pools[1])}")
    return pools


def sample_pool(pool, count, device):
    # Sampling with replacement avoids noisy tiny Attack batches.
    indices = torch.randint(len(pool), (count,))
    return pool[indices].to(device)


def train_hda_v4(source_model, teacher, source_loader, target_loader,
                 source_pools, target_pools, epochs=10, lr=1e-3,
                 class_batch_size=64, lambda_conditional=1.0):
    device = next(source_model.parameters()).device
    source_model.eval()
    teacher.eval()
    for model in (source_model, teacher):
        for parameter in model.parameters():
            parameter.requires_grad = False
    student = HDAV1Model(teacher.adapter.input_dim, source_model).to(device)
    student.adapter.load_state_dict(teacher.adapter.state_dict())
    optimizer = torch.optim.Adam(student.adapter.parameters(), lr=lr, weight_decay=1e-4)
    adapter_bns = [
        module for module in student.adapter.modules()
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
    ]
    history = []

    for epoch in range(1, epochs + 1):
        student.adapter.train()
        student.encoder.shared_fc2.eval()
        student.encoder.shared_bn2.eval()
        student.classifier.eval()
        source_batches = iter(source_loader)
        totals = {"loss": 0.0, "hidden_mmd": 0.0,
                  "normal_mmd": 0.0, "attack_mmd": 0.0}
        steps = 0
        for target_x in target_loader:
            try:
                source_x, _ = next(source_batches)
            except StopIteration:
                source_batches = iter(source_loader)
                try:
                    source_x, _ = next(source_batches)
                except StopIteration:
                    raise ValueError("Source marginal loader không có batch.") from None
            # Natural target batch updates TargetAdapter BN running statistics.
            with torch.no_grad():
                source_h = source_model.encode_hidden(source_x.to(device))
            target_h = student.adapter(target_x.to(device))
            hidden_mmd, _ = mmd_loss(source_h, target_h)

            # A separate balanced batch for conditional P(z | class).
            target_normal = sample_pool(target_pools[0], class_batch_size, device)
            target_attack = sample_pool(target_pools[1], class_batch_size, device)
            # Balanced conditional samples must not change running statistics.
            # eval mode still allows gradients through BN affine parameters.
            for bn in adapter_bns:
                bn.eval()
            try:
                target_z = student.encoder(
                    torch.cat((target_normal, target_attack), dim=0)
                )
            finally:
                for bn in adapter_bns:
                    bn.train()
            source_normal_z = sample_pool(source_pools[0], class_batch_size, device)
            source_attack_z = sample_pool(source_pools[1], class_batch_size, device)
            normal_mmd, _ = mmd_loss(source_normal_z, target_z[:class_batch_size])
            attack_mmd, _ = mmd_loss(source_attack_z, target_z[class_batch_size:])
            conditional_mmd = (normal_mmd + attack_mmd) / 2.0
            loss = hidden_mmd + lambda_conditional * conditional_mmd
            if not torch.isfinite(loss):
                raise ValueError("V4 MMD chứa NaN/Inf.")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            for name, value in (("loss", loss), ("hidden_mmd", hidden_mmd),
                                ("normal_mmd", normal_mmd), ("attack_mmd", attack_mmd)):
                totals[name] += value.item()
            steps += 1
        if steps == 0:
            raise ValueError("Target marginal loader không có batch.")
        result = {"epoch": epoch, **{name: value / steps for name, value in totals.items()}}
        history.append(result)
        print(f"Epoch {epoch:02d}/{epochs} | {result}")
    student.eval()
    return student, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--class-batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda-conditional", type=float, default=1.0)
    parser.add_argument("--protocol", default=None)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 2 or args.class_batch_size < 2:
        raise ValueError("epochs >= 1; batch sizes >= 2.")
    if not np.isfinite([args.lr, args.lambda_conditional]).all() or min(args.lr, args.lambda_conditional) <= 0:
        raise ValueError("lr và lambda-conditional phải dương, hữu hạn.")
    set_seed(args.seed)
    protocol = None
    protocol_hash = None
    target_train_path = FEATURE_DIR / "cicids_train"
    target_metadata_path = FEATURE_DIR / "cicids_metadata.json"
    checkpoint_dir = MODEL_DIR / "hda"
    if args.protocol is not None:
        from training.thesis_protocol import load_protocol, resolve_path, validate_training
        protocol, protocol_hash = load_protocol(args.protocol)
        validate_training(protocol, "v4", args.seed, args.epochs, args.batch_size,
                          args.lr, args.class_batch_size, args.lambda_conditional)
        target_train_path = resolve_path(protocol["target_data"]["adaptation_train"])
        target_metadata_path = resolve_path(protocol["target_data"]["metadata"])
        checkpoint_dir = resolve_path(protocol["checkpoint_dir"])
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    source_path = MODEL_DIR / "baselines" / f"unsw_seed{args.seed}.pt"
    teacher_path = checkpoint_dir / f"unsw_to_cicids_mmd_v2_seed{args.seed}.pt"
    source_checkpoint = torch.load(source_path, map_location="cpu", weights_only=True)
    teacher_checkpoint = torch.load(teacher_path, map_location="cpu", weights_only=True)
    if protocol is not None and teacher_checkpoint.get("protocol_sha256") != protocol_hash:
        raise ValueError("Teacher v2 chưa được retrain theo cùng frozen protocol.")
    if teacher_checkpoint.get("method") != "hda_shared_semantic_hidden_mmd":
        raise ValueError("Teacher phải là checkpoint HDA v2 hidden MMD.")
    source_dim = int(source_checkpoint["input_dim"])
    target_dim = int(teacher_checkpoint["target_dim"])
    if int(teacher_checkpoint["source_dim"]) != source_dim or int(teacher_checkpoint["seed"]) != args.seed:
        raise ValueError("Teacher không khớp source dimension/seed.")
    for domain, dimension in (("unsw", source_dim), ("cicids", target_dim)):
        metadata_path = target_metadata_path if domain == "cicids" else FEATURE_DIR / "unsw_metadata.json"
        metadata = json.loads(metadata_path.read_text())
        if int(metadata["input_dim"]) != dimension:
            raise ValueError(f"{domain} metadata không khớp checkpoint.")
    source = BaselineMLP(source_dim).to(device)
    source.load_state_dict(source_checkpoint["model_state_dict"])
    teacher = HDAV1Model(target_dim, source).to(device)
    teacher.adapter.load_state_dict(teacher_checkpoint["target_adapter_state_dict"])
    teacher.eval()
    print(f"V4 | device={device} | frozen teacher={teacher_path}")
    normal_pool, attack_pool, pseudo_metadata = build_pseudo_pools(
        teacher, make_teacher_loader(target_train_path, target_dim, args.batch_size), device
    )
    source_pools = build_source_pools(
        source, make_loader(FEATURE_DIR / "unsw_train", source_dim, args.batch_size), device
    )
    student, history = train_hda_v4(
        source, teacher,
        make_loader(FEATURE_DIR / "unsw_train", source_dim, args.batch_size, training=True),
        make_unlabeled_loader(target_train_path, target_dim, args.batch_size),
        source_pools, {0: normal_pool, 1: attack_pool},
        epochs=args.epochs, lr=args.lr, class_batch_size=args.class_batch_size,
        lambda_conditional=args.lambda_conditional,
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_path = checkpoint_dir / f"unsw_to_cicids_mmd_v4_seed{args.seed}.pt"
    checkpoint_data = {
        "method": "hda_fixed_v2_teacher_conditional_mmd", "version": "v4",
        "alignment_space": "hidden_256+conditional_latent_168",
        "seed": args.seed, "source_dim": source_dim, "target_dim": target_dim,
        "latent_dim": 168, "target_labels_used": False,
        "source_checkpoint": str(source_path), "teacher_checkpoint": str(teacher_path),
        "pseudo_label_metadata": pseudo_metadata, "lambda_conditional": args.lambda_conditional,
        "conditional_adapter_bn_mode": "eval",
        "epochs": args.epochs, "batch_size": args.batch_size,
        "class_batch_size": args.class_batch_size, "learning_rate": args.lr,
        "history": history,
        "target_adapter_state_dict": {k: v.detach().cpu() for k, v in student.adapter.state_dict().items()},
    }
    if protocol is not None:
        checkpoint_data["protocol_id"] = protocol["protocol_id"]
        checkpoint_data["protocol_sha256"] = protocol_hash
    torch.save(checkpoint_data, output_path)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
