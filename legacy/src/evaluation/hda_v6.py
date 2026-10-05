import argparse
import json

import numpy as np
import torch

from evaluation.baseline import compute_metrics, select_f1_threshold
from evaluation.hda import diagnose_pseudo_label_purity
from models.hda_v6 import HDAV6Model
from training.baseline import make_loader
from training.hda_v6 import ROOT, checkpoint_path, file_hash, load_checkpoint, load_config


def collect_predictions(model, loader, domain):
    model.eval()
    device = next(model.parameters()).device
    labels = []
    logits = []
    with torch.no_grad():
        for features, batch_labels in loader:
            _, batch_logits = model(features.to(device), domain)
            labels.append(batch_labels.cpu().numpy())
            logits.append(batch_logits.cpu())
    if not labels:
        raise ValueError(f"Empty {domain} evaluation loader")
    logits = torch.cat(logits)
    scores = torch.softmax(logits, dim=1)[:, 1].numpy()
    return np.concatenate(labels), scores, logits.numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/hda_v6.json")
    parser.add_argument("--version", choices=["v6a", "v6b", "v6c"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    config, config_hash = load_config(args.config)
    if args.seed not in config["development_seeds"]:
        raise ValueError("Seed is not declared for development")
    path = checkpoint_path(config, args.version, "adapt", args.seed)
    checkpoint = load_checkpoint(path, config_hash)
    if (checkpoint["version"] != args.version or checkpoint["seed"] != args.seed
            or checkpoint["stage"] != "adapt"):
        raise ValueError("Checkpoint version, seed or stage mismatch")
    model = HDAV6Model(checkpoint["source_dim"], checkpoint["target_dim"], checkpoint["architecture"])
    model.load_state_dict(checkpoint["model_state_dict"])
    source_loader = make_loader(
        ROOT / config["source_validation"], checkpoint["source_dim"], config["batch_size"],
    )
    source_labels, source_scores, _ = collect_predictions(model, source_loader, "source")
    if len(np.unique(source_labels)) != 2:
        raise ValueError("Source validation requires both classes")
    threshold = select_f1_threshold(source_labels, source_scores)
    target_loader = make_loader(
        ROOT / config["target_development"], checkpoint["target_dim"], config["batch_size"],
    )
    labels, scores, logits = collect_predictions(model, target_loader, "target")
    metrics = compute_metrics(labels, scores, threshold)
    diagnose_pseudo_label_purity(labels, logits)
    oracle_threshold = select_f1_threshold(labels, scores)
    oracle = compute_metrics(labels, scores, oracle_threshold)
    print(f"DEV ORACLE ONLY | threshold={oracle_threshold:.6f} | "
          f"F1={oracle['f1']:.6f} | Recall={oracle['recall']:.6f} | FPR={oracle['fpr']:.6f}")
    reference_path = ROOT / "configs/hda_v4_development_reference.json"
    reference = json.loads(reference_path.read_text())
    result = {
        "version": args.version, "seed": args.seed, "source_seed": config["source_seed"],
        "phase": "development", "target_labels_train": 0,
        "target_data": config["target_development"],
        "threshold_source": "UNSW validation after adaptation",
        "checkpoint_sha256": file_hash(path), "config_sha256": config_hash,
        "training_code_sha256": checkpoint["code_sha256"],
        "evaluation_code_sha256": {
            name: file_hash(ROOT / name) for name in (
                "src/evaluation/hda_v6.py", "src/evaluation/baseline.py", "src/evaluation/hda.py",
            )
        },
        "loss_weights": checkpoint["loss_weights"],
        "reference_sha256": file_hash(reference_path),
        "delta_vs_frozen_v4": {
            key: metrics[key] - value for key, value in reference["metrics"].items()
        },
        **metrics,
    }
    output = ROOT / config["result_dir"] / f"{args.version}_seed{args.seed}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"{args.version} | threshold={threshold:.6f} | AP={metrics['pr_auc']:.6f} | "
          f"ROC-AUC={metrics['roc_auc']:.6f} | F1={metrics['f1']:.6f} | "
          f"Recall={metrics['recall']:.6f} | FPR={metrics['fpr']:.6f}")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
