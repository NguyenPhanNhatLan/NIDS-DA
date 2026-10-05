"""Joint source/target UMAP after HDA, on development data only."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from umap import UMAP

from models.baseline import BaselineMLP
from models.hda_v1 import HDAV1Model
from training.adaptation import mmd_loss
from training.baseline import make_loader
from training.thesis_protocol import load_protocol, resolve_path, evaluation_target

PROJECT_DIR = Path(__file__).resolve().parents[2]


def sample_rows(loader, max_samples, seed):
    """Uniform random-priority sample across ALL rows, bounded CPU memory."""
    rng = np.random.default_rng(seed)
    selected_x = None
    selected_y = None
    priorities = np.empty(0)
    total = 0
    for features, labels in loader:
        total += len(labels)
        selected_x = features if selected_x is None else torch.cat((selected_x, features))
        selected_y = labels if selected_y is None else torch.cat((selected_y, labels))
        priorities = np.concatenate((priorities, rng.random(len(labels))))
        if len(priorities) > max_samples:
            keep = np.argpartition(priorities, max_samples - 1)[:max_samples]
            priorities = priorities[keep]
            selected_x = selected_x[keep]
            selected_y = selected_y[keep]
    if selected_x is None:
        raise ValueError("Loader rỗng.")
    print(f"Sampled {len(selected_y)}/{total} rows; attack rate={selected_y.float().mean():.4f}")
    return selected_x, selected_y.numpy()


def encode_rows(model, features, space, batch_size=256):
    model.eval()
    representations = []
    device = next(model.parameters()).device
    with torch.no_grad():
        for batch in features.split(batch_size):
            batch = batch.to(device)
            if space == "hidden":
                encoded = model.adapter(batch) if hasattr(model, "adapter") else model.encode_hidden(batch)
            else:
                encoded, _ = model(batch)
            representations.append(encoded.cpu())
    return torch.cat(representations)


def plot_umap(source_z, target_z, source_y, target_y, title, output,
              seed=42, n_neighbors=15, min_dist=0.1):
    # Fit ONCE on concatenated representations; never fit each domain separately.
    combined = torch.cat((source_z, target_z)).numpy()
    reducer = UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist,
                   metric="euclidean", random_state=seed, n_jobs=1)
    coordinates = reducer.fit_transform(combined)
    source_xy = coordinates[:len(source_z)]
    target_xy = coordinates[len(source_z):]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    axes[0].scatter(source_xy[:, 0], source_xy[:, 1], s=9, alpha=0.45,
                    color="tab:blue", label="UNSW")
    axes[0].scatter(target_xy[:, 0], target_xy[:, 1], s=9, alpha=0.45,
                    color="tab:orange", label="CICIDS")
    axes[0].set_title("Domains")
    axes[0].legend()
    for domain, xy, labels, marker in (
        ("UNSW", source_xy, source_y, "o"),
        ("CICIDS", target_xy, target_y, "^"),
    ):
        for label, name, color in ((0, "Normal", "tab:green"), (1, "Attack", "tab:red")):
            mask = labels == label
            axes[1].scatter(xy[mask, 0], xy[mask, 1], s=12, alpha=0.5,
                            marker=marker, color=color, label=f"{domain} {name}")
    axes[1].set_title("Classes (development labels only)")
    axes[1].legend()
    for axis in axes:
        axis.set_xlabel("UMAP 1")
        axis.set_ylabel("UMAP 2")
    fig.suptitle(title + "\nJoint UMAP; visual overlap does not prove alignment")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return coordinates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", choices=["v2", "v4"], default="v4")
    parser.add_argument("--adaptation-seed", "--seed", dest="seed", type=int, default=42)
    parser.add_argument("--protocol", default="configs/hda_thesis_protocol.json")
    parser.add_argument("--space", choices=["hidden", "latent"], default="latent")
    parser.add_argument("--max-samples", type=int, default=3000, help="Samples per domain.")
    parser.add_argument("--umap-seed", type=int, default=42)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.max_samples < 2 or args.n_neighbors < 2 or not 0 <= args.min_dist <= 1:
        raise ValueError("max-samples/n-neighbors >= 2; min-dist trong [0, 1].")
    protocol, protocol_hash = load_protocol(args.protocol)
    if args.seed not in protocol["development_seeds"]:
        raise ValueError("Adaptation seed không thuộc development protocol.")
    target_path = evaluation_target(protocol, "development")
    checkpoint_dir = resolve_path(protocol["checkpoint_dir"])
    checkpoint = torch.load(
        checkpoint_dir / f"unsw_to_cicids_mmd_{args.version}_seed{args.seed}.pt",
        map_location="cpu", weights_only=True,
    )
    expected_method = {"v2": "hda_shared_semantic_hidden_mmd",
                       "v4": "hda_fixed_v2_teacher_conditional_mmd"}[args.version]
    if checkpoint.get("protocol_sha256") != protocol_hash or checkpoint.get("method") != expected_method:
        raise ValueError("Checkpoint không khớp protocol/version.")
    source_seed = int(protocol["source_pretraining_seed"])
    if checkpoint.get("source_seed") != source_seed or checkpoint.get("adaptation_seed") != args.seed:
        raise ValueError("Source/adaptation seed không khớp checkpoint.")
    source_checkpoint = torch.load(
        PROJECT_DIR / "models/baselines" / f"unsw_seed{source_seed}.pt",
        map_location="cpu", weights_only=True,
    )
    source_dim = int(source_checkpoint["input_dim"])
    if int(checkpoint["source_dim"]) != source_dim:
        raise ValueError("Source dimension không khớp.")
    source = BaselineMLP(source_dim)
    source.load_state_dict(source_checkpoint["model_state_dict"])
    target = HDAV1Model(int(checkpoint["target_dim"]), source)
    target.adapter.load_state_dict(checkpoint["target_adapter_state_dict"])
    print(f"{args.version} | source seed={source_seed} | adaptation seed={args.seed} | {args.space}")
    print("Sampling UNSW test:")
    source_x, source_y = sample_rows(make_loader(
        PROJECT_DIR / "data/features/unsw_test", source_dim,
    ), args.max_samples, args.umap_seed)
    print("Sampling CICIDS development:")
    target_x, target_y = sample_rows(make_loader(
        target_path, int(checkpoint["target_dim"]),
    ), args.max_samples, args.umap_seed + 1)
    source_z = encode_rows(source, source_x, args.space)
    target_z = encode_rows(target, target_x, args.space)
    n = min(1000, len(source_z), len(target_z))
    with torch.no_grad():
        loss, bandwidth = mmd_loss(source_z[:n], target_z[:n])
    print(f"Original {source_z.shape[1]}D MMD² ({n}/domain): {loss.item():.6f}; bandwidth={bandwidth.item():.6f}")
    if args.n_neighbors >= len(source_z) + len(target_z):
        raise ValueError("n-neighbors phải nhỏ hơn tổng số mẫu.")
    output = args.output or (resolve_path(protocol["result_dir"]) / "development/umap" /
                             f"{args.version}_seed{args.seed}_{args.space}.png")
    plot_umap(source_z, target_z, source_y, target_y,
              f"HDA {args.version} | {args.space} {source_z.shape[1]}D | adaptation seed {args.seed}",
              output, args.umap_seed, args.n_neighbors, args.min_dist)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
