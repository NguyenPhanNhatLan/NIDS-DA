"""Shared HDAv1/v2 adapter training; MMD space is the only difference."""

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


def train_shared_hda(
    source_model, source_loader, target_loader, target_dim, epochs, lr,
    alignment_space="latent",
):
    if alignment_space not in ("latent", "hidden"):
        raise ValueError("alignment_space phải là latent hoặc hidden.")
    device = next(source_model.parameters()).device
    source_model.eval()
    for parameter in source_model.parameters():
        parameter.requires_grad = False

    target_model = HDAV1Model(target_dim, source_model).to(device)
    optimizer = torch.optim.Adam(
        target_model.adapter.parameters(), lr=lr, weight_decay=1e-4
    )
    history = []

    for epoch in range(1, epochs + 1):
        target_model.adapter.train()
        target_model.encoder.shared_fc2.eval()
        target_model.encoder.shared_bn2.eval()
        target_model.classifier.eval()
        source_batches = iter(source_loader)
        total_mmd = 0.0
        total_bandwidth = 0.0
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
            if alignment_space == "hidden":
                with torch.no_grad():
                    source_features = source_model.encode_hidden(source_x)
                target_features = target_model.adapter(target_x)
            else:
                with torch.no_grad():
                    source_features, _ = source_model(source_x)
                target_features = target_model.encoder(target_x)

            loss, bandwidth = mmd_loss(source_features, target_features)
            if not torch.isfinite(loss):
                raise ValueError(f"{alignment_space} MMD contains NaN/Inf.")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_mmd += loss.item()
            total_bandwidth += bandwidth.item()
            steps += 1

        if steps == 0:
            raise ValueError("CICIDS train không đủ một batch.")
        result = {
            "epoch": epoch,
            "mmd": total_mmd / steps,
            "bandwidth": total_bandwidth / steps,
        }
        history.append(result)
        print(
            f"Epoch {epoch:02d}/{epochs} | MMD={result['mmd']:.6f} | "
            f"bandwidth={result['bandwidth']:.4f}"
        )

    target_model.eval()
    return target_model, history


def run_training(seed=42, epochs=10, batch_size=256, lr=1e-3, alignment_space="latent"):
    if batch_size < 2 or epochs < 1 or lr <= 0:
        raise ValueError("batch-size >= 2, epochs >= 1 và lr > 0 là bắt buộc")
    if alignment_space not in ("latent", "hidden"):
        raise ValueError("alignment_space phải là latent hoặc hidden.")
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
    target_model, history = train_shared_hda(
        source_model, source_loader, target_loader, target_dim,
        epochs=epochs, lr=lr, alignment_space=alignment_space,
    )

    output_dir = MODEL_DIR / "hda"
    output_dir.mkdir(parents=True, exist_ok=True)
    version = "v2" if alignment_space == "hidden" else "v1"
    output_path = output_dir / f"unsw_to_cicids_mmd_{version}_seed{seed}.pt"
    torch.save(
        {
            "method": (
                "hda_shared_semantic_hidden_mmd"
                if version == "v2" else "hda_v1_shared_tail_single_rbf_mmd"
            ),
            "version": version,
            "alignment_space": (
                "hidden_256" if version == "v2" else "latent_168"
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
        },
        output_path,
    )
    print(f"Saved: {output_path}")

