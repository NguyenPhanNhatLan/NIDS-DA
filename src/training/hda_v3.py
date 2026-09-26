"""HDAv3: align 256D hidden and 168D latent distributions."""

import argparse

from training.hda_shared import run_training, train_shared_hda


def train_hda_v3(source_model, source_loader, target_loader, target_dim,
                 epochs, lr, lambda_latent=None):
    return train_shared_hda(
        source_model, source_loader, target_loader, target_dim,
        epochs=epochs, lr=lr, alignment_space="dual",
        lambda_latent=lambda_latent,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--lambda-latent", type=float, default=None,
        help="Default: ratio of initial hidden/latent MMD over three batches.",
    )
    args = parser.parse_args()
    run_training(
        seed=args.seed, epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, alignment_space="dual", lambda_latent=args.lambda_latent,
    )


if __name__ == "__main__":
    main()
