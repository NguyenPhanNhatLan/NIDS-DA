"""
Heterogeneous Unsupervised Domain Adaptation
UNSW-NB15 -> CICIDS2017

HDA v0:
- Source model: pretrained UNSW MLP
- Source encoder: frozen
- Source classifier: frozen
- Target encoder: trainable
- Target labels: NOT loaded
- Alignment: marginal single-RBF MMD
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import (
    DataLoader,
    IterableDataset,
)

from models.baseline import BaselineMLP
from models.target_encoder import (
    TargetEncoder,
    TargetModel,
)

from training.baseline import (
    make_loader,
    set_seed,
)


PROJECT_DIR = Path(__file__).resolve().parents[2]

FEATURE_DIR = PROJECT_DIR / "data" / "features"
MODEL_DIR = PROJECT_DIR / "models"


# ==========================================================
# Target loader
# ==========================================================

class UnlabeledParquetRows(IterableDataset):
    """
    Chỉ đọc feature của target.

    Quan trọng:
    Không đọc label của CICIDS trong adaptation.
    """

    def __init__(
        self,
        path,
        input_dim,
        training=True,
    ):
        self.files = sorted(
            Path(path).glob("*.parquet")
        )

        if not self.files:
            raise FileNotFoundError(
                f"No parquet files in {path}"
            )

        self.input_dim = input_dim
        self.training = training

    def __iter__(self):

        files = list(self.files)

        seed = torch.randint(
            0,
            2**32,
            (),
        ).item()

        rng = np.random.default_rng(seed)

        if self.training:
            rng.shuffle(files)

        for path in files:

            with pq.ParquetFile(path) as parquet:

                # CHỈ ĐỌC FEATURES
                batches = parquet.iter_batches(
                    batch_size=8192,
                    columns=["features"],
                )

                for batch in batches:

                    feature_rows = (
                        batch
                        .column("features")
                        .to_pylist()
                    )

                    features = np.asarray(
                        feature_rows,
                        dtype=np.float32,
                    )

                    if (
                        features.ndim != 2
                        or
                        features.shape[1]
                        != self.input_dim
                    ):
                        raise ValueError(
                            f"Wrong target feature "
                            f"dimension in {path}"
                        )

                    if not np.isfinite(
                        features
                    ).all():
                        raise ValueError(
                            f"Target contains "
                            f"NaN/Inf: {path}"
                        )

                    indices = np.arange(
                        len(features)
                    )

                    if self.training:
                        rng.shuffle(indices)

                    features = torch.from_numpy(
                        features
                    )

                    for index in indices:
                        yield features[index]


def make_unlabeled_loader(
    path,
    input_dim,
    batch_size=256,
):
    dataset = UnlabeledParquetRows(
        path,
        input_dim,
        training=True,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,

        # BatchNorm + MMD đều dễ xử lý hơn
        # khi bỏ batch cuối quá nhỏ.
        drop_last=True,
    )


# ==========================================================
# MMD
# ==========================================================

def estimate_bandwidth_squared(
    source_features,
    target_features,
):
    """
    Median heuristic.

    Dùng cả source và target representation.
    Không dùng label.
    """

    with torch.no_grad():

        combined = torch.cat(
            [
                source_features.detach(),
                target_features.detach(),
            ],
            dim=0,
        )

        distance_squared = torch.cdist(
            combined,
            combined,
        ).square()

        positive = distance_squared[
            distance_squared > 1e-12
        ]

        if positive.numel() == 0:
            return torch.tensor(
                1.0,
                device=combined.device,
            )

        bandwidth_squared = torch.median(
            positive
        )

        return bandwidth_squared.clamp_min(
            1e-6
        )


def rbf_kernel(
    x,
    y,
    bandwidth_squared,
):
    distance_squared = torch.cdist(
        x,
        y,
    ).square()

    return torch.exp(
        -distance_squared
        /
        (2.0 * bandwidth_squared)
    )


def mmd_loss(
    source_features,
    target_features,
):
    """
    Biased empirical MMD^2.

    MMD^2 =
        E[k(xs, xs')]
        + E[k(xt, xt')]
        - 2 E[k(xs, xt)]
    """

    batch_size = min(
        len(source_features),
        len(target_features),
    )

    source_features = (
        source_features[:batch_size]
    )

    target_features = (
        target_features[:batch_size]
    )

    bandwidth_squared = (
        estimate_bandwidth_squared(
            source_features,
            target_features,
        )
    )

    source_kernel = rbf_kernel(
        source_features,
        source_features,
        bandwidth_squared,
    )

    target_kernel = rbf_kernel(
        target_features,
        target_features,
        bandwidth_squared,
    )

    cross_kernel = rbf_kernel(
        source_features,
        target_features,
        bandwidth_squared,
    )

    loss = (
        source_kernel.mean()
        + target_kernel.mean()
        - 2.0 * cross_kernel.mean()
    )

    return (
        loss,
        torch.sqrt(
            bandwidth_squared
        ),
    )


# ==========================================================
# HDA training
# ==========================================================

def train_adaptation(
    source_model,
    source_loader,
    target_loader,
    target_dim,
    epochs=10,
    learning_rate=1e-3,
):

    device = next(
        source_model.parameters()
    ).device

    # ------------------------------------------------------
    # Freeze entire source model
    # ------------------------------------------------------

    source_model.eval()

    for parameter in source_model.parameters():
        parameter.requires_grad = False

    # ------------------------------------------------------
    # Create target encoder
    # ------------------------------------------------------

    target_encoder = TargetEncoder(
        input_dim=target_dim,
        latent_dim=168,
    )

    target_model = TargetModel(
        target_encoder,
        source_model.classifier,
    ).to(device)

    optimizer = torch.optim.Adam(
        target_model.encoder.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )

    history = []

    # ======================================================
    # Training
    # ======================================================

    for epoch in range(
        1,
        epochs + 1,
    ):

        target_model.encoder.train()
        target_model.classifier.eval()

        source_iterator = iter(
            source_loader
        )

        total_mmd = 0.0
        total_bandwidth = 0.0
        steps = 0

        for target_x in target_loader:

            # ----------------------------------------------
            # Get source batch
            # ----------------------------------------------

            try:
                source_x, _ = next(
                    source_iterator
                )

            except StopIteration:

                source_iterator = iter(
                    source_loader
                )

                source_x, _ = next(
                    source_iterator
                )

            source_x = source_x.to(
                device
            )

            target_x = target_x.to(
                device
            )

            # ----------------------------------------------
            # Source representation
            # ----------------------------------------------

            with torch.no_grad():

                source_features, _ = (
                    source_model(
                        source_x
                    )
                )

            # ----------------------------------------------
            # Target representation
            # ----------------------------------------------

            target_features = (
                target_model.encoder(
                    target_x
                )
            )

            # ----------------------------------------------
            # MMD
            # ----------------------------------------------

            loss, bandwidth = mmd_loss(
                source_features,
                target_features,
            )

            if not torch.isfinite(loss):
                raise ValueError(
                    "MMD loss contains NaN/Inf."
                )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_mmd += loss.item()
            total_bandwidth += (
                bandwidth.item()
            )

            steps += 1

        if steps == 0:
            raise RuntimeError(
                "No adaptation batches."
            )

        mean_mmd = (
            total_mmd / steps
        )

        mean_bandwidth = (
            total_bandwidth / steps
        )

        history.append(
            {
                "epoch": epoch,
                "mmd": mean_mmd,
                "bandwidth":
                    mean_bandwidth,
            }
        )

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"MMD={mean_mmd:.6f} | "
            f"bandwidth="
            f"{mean_bandwidth:.4f}"
        )

    target_model.eval()

    return target_model, history


# ==========================================================
# CLI
# ==========================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
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

    args = parser.parse_args()

    set_seed(args.seed)

    # ======================================================
    # Metadata
    # ======================================================

    with (
        FEATURE_DIR
        / "unsw_metadata.json"
    ).open(
        encoding="utf-8"
    ) as file:

        source_metadata = json.load(
            file
        )

    with (
        FEATURE_DIR
        / "cicids_metadata.json"
    ).open(
        encoding="utf-8"
    ) as file:

        target_metadata = json.load(
            file
        )

    source_dim = int(
        source_metadata["input_dim"]
    )

    target_dim = int(
        target_metadata["input_dim"]
    )

    print(
        f"Source dim: {source_dim}"
    )

    print(
        f"Target dim: {target_dim}"
    )

    print(
        "Latent dim: 168"
    )


    if torch.cuda.is_available():
        device = torch.device("cuda")

    elif torch.backends.mps.is_available():
        device = torch.device("mps")

    else:
        device = torch.device("cpu")

    print(
        f"Device: {device}"
    )
    
    
    source_checkpoint_path = (
        MODEL_DIR
        / "baselines"
        / f"unsw_seed{args.seed}.pt"
    )

    checkpoint = torch.load(
        source_checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    source_model = BaselineMLP(
        input_dim=source_dim
    )

    source_model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    source_model = source_model.to(
        device
    )

    # ======================================================
    # Data loaders
    # ======================================================

    source_loader = make_loader(
        FEATURE_DIR / "unsw_train",
        input_dim=source_dim,
        batch_size=args.batch_size,
        training=True,
    )

    target_loader = (
        make_unlabeled_loader(
            FEATURE_DIR
            / "cicids_train",
            input_dim=target_dim,
            batch_size=args.batch_size,
        )
    )

    # ======================================================
    # Train HDA
    # ======================================================

    target_model, history = (
        train_adaptation(
            source_model,
            source_loader,
            target_loader,
            target_dim=target_dim,
            epochs=args.epochs,
            learning_rate=args.lr,
        )
    )

    # ======================================================
    # Save
    # ======================================================

    output_dir = (
        MODEL_DIR
        / "hda"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / (
            "unsw_to_cicids_"
            f"mmd_seed{args.seed}.pt"
        )
    )

    # Move encoder to CPU before saving.
    target_model.encoder.cpu()

    torch.save(
        {
            "method":
                "hda_marginal_rbf_mmd",

            "source_dataset":
                "unsw",

            "target_dataset":
                "cicids",

            "seed":
                args.seed,

            "source_dim":
                source_dim,

            "target_dim":
                target_dim,

            "latent_dim":
                168,

            "epochs":
                args.epochs,

            "batch_size":
                args.batch_size,

            "learning_rate":
                args.lr,

            "target_labels_used":
                False,

            "source_checkpoint":
                str(
                    source_checkpoint_path
                ),

            "history":
                history,

            "target_encoder_state_dict":
                target_model
                .encoder
                .state_dict(),
        },
        output_path,
    )

    print(
        f"\nSaved HDA model:\n"
        f"{output_path}"
    )


if __name__ == "__main__":
    main()