"""HDAv1: align CICIDS and UNSW at the shared 168D latent space."""

import argparse

from training.hda_shared import run_training, train_shared_hda as train_hda_v1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    run_training(
        seed=args.seed, epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, alignment_space="latent",
    )


if __name__ == "__main__":
    main()
