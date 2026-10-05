"""Aggregate v2/v4 adaptation seeds with a fixed UNSW source seed."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
RESULT_DIR = PROJECT_DIR / "results/thesis/hda_thesis_v2_fixed_source/development/hda"
SEEDS = (42, 43, 44)
EXPERIMENTS = {
    "v2": "HDA shared tail + hidden RBF-MMD",
    "v4": "HDA shared tail + fixed v2 pseudo-label conditional RBF-MMD",
}
METRICS = {
    "AP": "pr_auc",
    "ROC-AUC": "roc_auc",
    "F1": "f1",
    "Recall": "recall",
    "FPR": "fpr",
}


def aggregate_results(result_dir=RESULT_DIR):
    rows = []
    reference = None
    for version, experiment in EXPERIMENTS.items():
        results = []
        for seed in SEEDS:
            path = Path(result_dir) / f"unsw_to_cicids_mmd_{version}_seed{seed}.json"
            if not path.exists():
                raise FileNotFoundError(f"Thiếu run bắt buộc: {path}")
            with path.open(encoding="utf-8") as file:
                result = json.load(file)
            expected = {
                "version": version, "experiment": experiment,
                "seed": seed, "adaptation_seed": seed, "source_seed": 42,
                "protocol_id": "hda_thesis_v2_fixed_source",
                "phase": "development", "target_labels_train": 0,
                "source_domain": "UNSW", "target_domain": "CICIDS",
            }
            for name, value in expected.items():
                if result.get(name) != value:
                    raise ValueError(f"{path}: {name} phải là {value!r}.")
            if not result.get("protocol_sha256") or not result.get("target_data"):
                raise ValueError(f"{path}: thiếu protocol hash hoặc target_data.")
            signature = (
                result["protocol_sha256"], result["target_data"],
                result["threshold_source"],
                result["tn"] + result["fp"], result["fn"] + result["tp"],
            )
            if reference is None:
                reference = signature
            elif signature != reference:
                raise ValueError(f"{path}: protocol, tập đánh giá hoặc threshold khác các run còn lại.")
            for key in METRICS.values():
                value = float(result[key])
                if not np.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError(f"{path}: {key} không hợp lệ.")
            results.append(result)

        row = {"Version": version, "Source seed": 42, "N": len(results)}
        for title, key in METRICS.items():
            values = np.array([result[key] for result in results], dtype=float)
            row[f"{title} mean"] = values.mean()
            row[f"{title} std"] = values.std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--output", type=Path, default=None, help="Optional numeric CSV.")
    args = parser.parse_args()
    summary = aggregate_results(args.result_dir)
    display = summary[["Version", "Source seed", "N"]].copy()
    for title in METRICS:
        display[title] = [
            f"{mean:.6f} ± {std:.6f}"
            for mean, std in zip(summary[f"{title} mean"], summary[f"{title} std"])
        ]
    print("Development/internal evaluation | adaptation seeds: 42, 43, 44")
    print("Fixed source seed: 42 | mean ± sample std (ddof=1)")
    print(display.to_string(index=False))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.output, index=False)
        print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
