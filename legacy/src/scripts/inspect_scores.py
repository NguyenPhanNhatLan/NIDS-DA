import json
from pathlib import Path

import numpy as np
import torch

from evaluation.baseline import collect_scores, compute_metrics, select_f1_threshold
from features.common_features import COMMON_FEATURES
from models.baseline import BaselineMLP
from training.proposal_data import ParquetBatchStream
from training.proposal_mmd import output_paths

ROOT = Path.cwd()
DIRECTION = "cicids_to_unsw"
SEED = 42
CONFIG = ROOT / "configs/proposal_mmd_lambda0001.json"

checkpoint_path, result_path = output_paths(DIRECTION, SEED, CONFIG)
result = json.loads(result_path.read_text())

assert result["direction"] == DIRECTION
assert result["seed"] == SEED
assert result["lambda_mmd"] == 0.001
assert result["target_development_split"] == "unsw_val"
assert result["checkpoint"] == str(checkpoint_path)

checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
model = BaselineMLP(len(COMMON_FEATURES))
model.load_state_dict(checkpoint["model_state_dict"])

loader = ParquetBatchStream(
    ROOT / "data/features/proposal_v2/cicids_to_unsw/unsw_val",
    batch_size=1024,
    shuffle=False,
    seed=SEED,
    include_labels=True,
)
labels, scores = collect_scores(model, loader)

print("Statistic       Normal        Attack")
for name, fn in [
    ("mean", np.mean),
    ("median", np.median),
    ("q05", lambda x: np.quantile(x, 0.05)),
    ("q25", lambda x: np.quantile(x, 0.25)),
    ("q75", lambda x: np.quantile(x, 0.75)),
    ("q95", lambda x: np.quantile(x, 0.95)),
]:
    print(
        f"{name:10s} {fn(scores[labels == 0]):12.6f} "
        f"{fn(scores[labels == 1]):12.6f}"
    )

source_threshold = result["threshold"]
print(f"\nSource threshold:      {source_threshold:.6f}")
print(f"Predicted attack rate: {np.mean(scores >= source_threshold):.6f}")
print(f"Target prevalence:     {np.mean(labels == 1):.6f}")

source_metrics = compute_metrics(labels, scores, source_threshold)
print(f"Recall at source threshold: {source_metrics['recall']:.6f}")
print(f"F1 at source threshold:     {source_metrics['f1']:.6f}")


oracle_threshold = select_f1_threshold(labels, scores)
oracle_metrics = compute_metrics(labels, scores, oracle_threshold)
print(f"\nDiagnostic target oracle threshold: {oracle_threshold:.6f}")
print(f"Diagnostic oracle F1:               {oracle_metrics['f1']:.6f}")
