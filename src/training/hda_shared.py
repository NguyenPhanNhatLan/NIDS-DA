"""Shared HDA adapter training with latent, hidden, or dual MMD."""

import json
from pathlib import Path

import torch

from models.baseline import BaselineMLP
from models.hda_v1 import HDAV1Model
from training.adaptation import make_unlabeled_loader, mmd_loss
from training.baseline import make_loader, set_seed

PROJECT_DIR = Path(__file__).resolve().parents[2]
FEATURE_DIR = PROJECT_DIR / "data" / "features"
MODEL_DIR = PROJECT_DIR / "models"


def dual_mmd(source_model, target_model, source_x, target_x):
    with torch.no_grad():
        source_h = source_model.encode_hidden(source_x)
        source_z, _ = source_model(source_x)

    target_h = target_model.adapter(target_x)
    target_z = target_model.encoder.shared_fc2(target_h)
    target_z = target_model.encoder.shared_bn2(target_z)
    target_z = torch.relu(target_z)

    hidden_mmd, hidden_bw = mmd_loss(source_h, target_h)
    latent_mmd, latent_bw = mmd_loss(source_z, target_z)
    return hidden_mmd, hidden_bw, latent_mmd, latent_bw


def initial_dual_mmd(source_model, target_model, source_loader, target_loader, device):
    """Measure three batches without labels or running-stat updates."""
    target_model.eval()
    source_batches = iter(source_loader)
    hidden_values = []
    latent_values = []
    with torch.no_grad():
        for step, target_x in enumerate(target_loader):
            if step == 3:
                break
            try:
                source_x, _ = next(source_batches)
            except StopIteration:
                source_batches = iter(source_loader)
                source_x, _ = next(source_batches)
            hidden_mmd, _, latent_mmd, _ = dual_mmd(
                source_model, target_model,
                source_x.to(device), target_x.to(device),
            )
            hidden_values.append(hidden_mmd.item())
            latent_values.append(latent_mmd.item())

    if not hidden_values:
        raise ValueError("Không có batch để đo MMD ban đầu.")
    hidden_initial = sum(hidden_values) / len(hidden_values)
    latent_initial = sum(latent_values) / len(latent_values)
    if not torch.isfinite(torch.tensor([hidden_initial, latent_initial])).all():
        raise ValueError("MMD ban đầu chứa NaN/Inf.")
    if hidden_initial <= 1e-8:
        raise ValueError("Hidden MMD ban đầu gần 0; không thể tính lambda_latent.")
    if latent_initial <= 1e-8:
        raise ValueError("Latent MMD ban đầu gần 0; không thể tính lambda_latent.")
    lambda_latent = hidden_initial / latent_initial
    print(
        f"Initial ({len(hidden_values)} batches): hidden MMD={hidden_initial:.6f} | "
        f"latent MMD={latent_initial:.6f} | lambda_latent={lambda_latent:.6f}"
    )
    return lambda_latent, hidden_initial, latent_initial


def train_shared_hda(
    source_model, source_loader, target_loader, target_dim, epochs, lr,
    alignment_space="latent", lambda_latent=None,
):
    if alignment_space not in ("latent", "hidden", "dual"):
        raise ValueError("alignment_space phải là latent, hidden hoặc dual.")
    if lambda_latent is not None and (not torch.isfinite(torch.tensor(lambda_latent)) or lambda_latent <= 0):
        raise ValueError("lambda_latent phải là số dương hữu hạn.")
    device = next(source_model.parameters()).device
    source_model.eval()
    for parameter in source_model.parameters():
        parameter.requires_grad = False

    target_model = HDAV1Model(target_dim, source_model).to(device)
    optimizer = torch.optim.Adam(
        target_model.adapter.parameters(), lr=lr, weight_decay=1e-4
    )
    history = []
    initial_mmd = None
    if alignment_space == "dual":
        measured_lambda, hidden_initial, latent_initial = initial_dual_mmd(
            source_model, target_model, source_loader, target_loader, device
        )
        initial_mmd = {"hidden": hidden_initial, "latent": latent_initial}
        if lambda_latent is None:
            lambda_latent = measured_lambda
        else:
            print(f"Fixed lambda_latent override: {lambda_latent:.6f}")

    for epoch in range(1, epochs + 1):
        target_model.adapter.train()
        target_model.encoder.shared_fc2.eval()
        target_model.encoder.shared_bn2.eval()
        target_model.classifier.eval()
        source_batches = iter(source_loader)
        total_mmd = 0.0
        total_bandwidth = 0.0
        total_hidden_mmd = 0.0
        total_latent_mmd = 0.0
        total_hidden_bw = 0.0
        total_latent_bw = 0.0
        steps = 0

        for target_x in target_loader:
            try:
                source_x, _ = next(source_batches)
            except StopIteration:
                source_batches = iter(source_loader)
                try:
                    source_x, _ = next(source_batches)
                except StopIteration:
                    raise ValueError("UNSW train không đủ một batch.") from None

            source_x = source_x.to(device)
            target_x = target_x.to(device)
            if alignment_space == "dual":
                hidden_mmd, hidden_bw, latent_mmd, latent_bw = dual_mmd(
                    source_model, target_model, source_x, target_x
                )
                loss = hidden_mmd + lambda_latent * latent_mmd
                total_hidden_mmd += hidden_mmd.item()
                total_latent_mmd += latent_mmd.item()
                total_hidden_bw += hidden_bw.item()
                total_latent_bw += latent_bw.item()
            elif alignment_space == "hidden":
                with torch.no_grad():
                    source_features = source_model.encode_hidden(source_x)
                target_features = target_model.adapter(target_x)
            else:
                with torch.no_grad():
                    source_features, _ = source_model(source_x)
                target_features = target_model.encoder(target_x)

            if alignment_space != "dual":
                loss, bandwidth = mmd_loss(source_features, target_features)
            if not torch.isfinite(loss):
                raise ValueError(f"{alignment_space} MMD contains NaN/Inf.")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_mmd += loss.item()
            if alignment_space != "dual":
                total_bandwidth += bandwidth.item()
            steps += 1

        if steps == 0:
            raise ValueError("CICIDS train không đủ một batch.")
        result = {"epoch": epoch, "mmd": total_mmd / steps}
        if alignment_space == "dual":
            result.update({
                "hidden_mmd": total_hidden_mmd / steps,
                "latent_mmd": total_latent_mmd / steps,
                "hidden_bandwidth": total_hidden_bw / steps,
                "latent_bandwidth": total_latent_bw / steps,
            })
        else:
            result["bandwidth"] = total_bandwidth / steps
        history.append(result)
        if alignment_space == "dual":
            print(
                f"Epoch {epoch:02d}/{epochs} | loss={result['mmd']:.6f} | "
                f"hidden={result['hidden_mmd']:.6f} | latent={result['latent_mmd']:.6f}"
            )
        else:
            print(
                f"Epoch {epoch:02d}/{epochs} | MMD={result['mmd']:.6f} | "
                f"bandwidth={result['bandwidth']:.4f}"
            )

    target_model.eval()
    if alignment_space == "dual":
        return target_model, history, lambda_latent, initial_mmd
    return target_model, history


def run_training(seed=42, epochs=10, batch_size=256, lr=1e-3,
                 alignment_space="latent", lambda_latent=None):
    if batch_size < 2 or epochs < 1 or lr <= 0:
        raise ValueError("batch-size >= 2, epochs >= 1 và lr > 0 là bắt buộc")
    if alignment_space not in ("latent", "hidden", "dual"):
        raise ValueError("alignment_space phải là latent, hidden hoặc dual.")
    set_seed(seed)

    with (FEATURE_DIR / "unsw_metadata.json").open(encoding="utf-8") as file:
        source_dim = int(json.load(file)["input_dim"])
    with (FEATURE_DIR / "cicids_metadata.json").open(encoding="utf-8") as file:
        target_dim = int(json.load(file)["input_dim"])

    checkpoint_path = MODEL_DIR / "baselines" / f"unsw_seed{seed}.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if int(checkpoint["input_dim"]) != source_dim:
        raise ValueError("UNSW checkpoint không khớp metadata.")
    source_model = BaselineMLP(source_dim)
    source_model.load_state_dict(checkpoint["model_state_dict"])
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    source_model.to(device)
    print(f"Source {source_dim} -> 256 -> 168 | Target {target_dim} -> 256 -> 168")
    print(f"Device: {device} | seed: {seed} | MMD: {alignment_space}")

    source_loader = make_loader(
        FEATURE_DIR / "unsw_train", source_dim, batch_size, training=True
    )
    # Target labels are never loaded by this loader.
    target_loader = make_unlabeled_loader(
        FEATURE_DIR / "cicids_train", target_dim, batch_size
    )
    training_result = train_shared_hda(
        source_model, source_loader, target_loader, target_dim,
        epochs=epochs, lr=lr, alignment_space=alignment_space,
        lambda_latent=lambda_latent,
    )
    target_model, history = training_result[:2]
    if alignment_space == "dual":
        lambda_latent, initial_mmd = training_result[2:]

    output_dir = MODEL_DIR / "hda"
    output_dir.mkdir(parents=True, exist_ok=True)
    version = {"latent": "v1", "hidden": "v2", "dual": "v3"}[alignment_space]
    output_path = output_dir / f"unsw_to_cicids_mmd_{version}_seed{seed}.pt"
    checkpoint_data = {
            "method": (
                {"v1": "hda_v1_shared_tail_single_rbf_mmd",
                 "v2": "hda_shared_semantic_hidden_mmd",
                 "v3": "hda_dual_level_single_rbf_mmd"}[version]
            ),
            "version": version,
            "alignment_space": (
                {"v1": "latent_168", "v2": "hidden_256",
                 "v3": "hidden_256+latent_168"}[version]
            ),
            "seed": seed,
            "source_dim": source_dim,
            "target_dim": target_dim,
            "latent_dim": 168,
            "target_labels_used": False,
            "source_checkpoint": str(checkpoint_path),
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "history": history,
            "target_adapter_state_dict": {
                key: value.detach().cpu()
                for key, value in target_model.adapter.state_dict().items()
            },
        }
    if version == "v3":
        checkpoint_data["lambda_latent"] = lambda_latent
        checkpoint_data["initial_mmd"] = initial_mmd
    torch.save(checkpoint_data, output_path)
    print(f"Saved: {output_path}")
