"""Auxiliary source-only LR/XGBoost baselines on prepared proposal_v1 features."""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression

from evaluation.baseline import compute_metrics, select_f1_threshold
from experiments.proposal_source_only import domains, sha256
from features.common_features import COMMON_FEATURES
from training.proposal_data import split_sha256

ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = ROOT / "results/proposal_v1/classical_target_val"
COMMON_CONFIG = ROOT / "configs/common_features_v1.json"
INPUT_DIM = len(COMMON_FEATURES)


def read_batches(directory, batch_size=8192):
    files = sorted(Path(directory).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet files in {directory}")
    for file in files:
        with pq.ParquetFile(file) as parquet:
            for batch in parquet.iter_batches(batch_size=batch_size, columns=["features", "label"]):
                vectors = batch.column("features")
                if not pa.types.is_fixed_size_list(vectors.type) or vectors.type.list_size != INPUT_DIM:
                    raise ValueError(f"Expected {INPUT_DIM} features in {file}")
                if vectors.null_count or vectors.values.null_count:
                    raise ValueError(f"Null features in {file}")
                x = np.asarray(vectors.values.to_numpy(zero_copy_only=False), dtype=np.float32)
                x = x.reshape(len(batch), INPUT_DIM)
                y = np.asarray(batch.column("label").to_numpy(zero_copy_only=False), dtype=np.int64)
                if not np.isfinite(x).all() or not np.isin(y, [0, 1]).all():
                    raise ValueError(f"Invalid features or labels in {file}")
                yield x, y


def sample_source_train(directory, max_rows, seed):
    if max_rows < 2:
        raise ValueError("max_rows must be at least 2")
    files = sorted(Path(directory).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet files in {directory}")
    total_rows = sum(pq.read_metadata(file).num_rows for file in files)
    chosen = np.sort(np.random.default_rng(seed).choice(
        total_rows, size=min(total_rows, max_rows), replace=False
    ))
    selected_x, selected_y = [], []
    seen = 0
    for x, y in read_batches(directory):
        start = np.searchsorted(chosen, seen, side="left")
        end = np.searchsorted(chosen, seen + len(x), side="left")
        if end > start:
            local = chosen[start:end] - seen
            selected_x.append(x[local])
            selected_y.append(y[local])
        seen += len(x)
    if not selected_x:
        raise ValueError(f"Empty source training split: {directory}")
    features = np.concatenate(selected_x)
    labels = np.concatenate(selected_y)
    if len(np.unique(labels)) != 2:
        raise ValueError("Source training sample needs both classes")
    return features, labels, total_rows


def build_model(method, seed, labels):
    if method == "logistic_regression":
        return LogisticRegression(max_iter=300, class_weight="balanced", random_state=seed)
    if method == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as error:
            raise ImportError("XGBoost baseline requires: .venv/bin/pip install '.[classical]'") from error
        negatives = int((labels == 0).sum())
        positives = int((labels == 1).sum())
        return XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            tree_method="hist", n_jobs=4, random_state=seed,
            scale_pos_weight=negatives / positives, eval_metric="logloss",
        )
    raise ValueError(f"Unknown classical method: {method}")


def score_split(model, directory):
    labels, scores = [], []
    for x, y in read_batches(directory):
        labels.append(y)
        scores.append(model.predict_proba(x)[:, 1])
    return np.concatenate(labels), np.concatenate(scores)


def run(direction, method, seed=42, max_train_rows=250000):
    source, target = domains(direction)
    if method not in {"logistic_regression", "xgboost"}:
        raise ValueError(f"Unknown classical method: {method}")
    base = ROOT / "data/features/proposal_v1" / direction
    source_train = base / f"{source}_train"
    source_val = base / f"{source}_val"
    target_val = base / f"{target}_val"
    preprocessor = ROOT / "models/proposal_v1" / direction / "preprocessor.joblib"
    if not preprocessor.is_file():
        raise FileNotFoundError(f"Missing source-fitted preprocessor: {preprocessor}")
    output = RESULT_ROOT / method / direction / f"seed{seed}.json"
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")

    x_train, y_train, total_rows = sample_source_train(source_train, max_train_rows, seed)
    model = build_model(method, seed, y_train)
    model.fit(x_train, y_train)
    source_y, source_scores = score_split(model, source_val)
    threshold = select_f1_threshold(source_y, source_scores)
    target_y, target_scores = score_split(model, target_val)
    result = {
        "protocol": "proposal_classical_target_val_v1",
        "method": method,
        "direction": direction,
        "seed": seed,
        "features": list(COMMON_FEATURES),
        "feature_count": INPUT_DIM,
        "train_rows_total": total_rows,
        "train_rows_sampled": len(y_train),
        "max_train_rows": max_train_rows,
        "threshold_from_source_val": threshold,
        "source_val": compute_metrics(source_y, source_scores, threshold),
        "target_development_split": f"{target}_val",
        "target_development": compute_metrics(target_y, target_scores, threshold),
        "common_feature_config_sha256": sha256(COMMON_CONFIG),
        "preprocessor_sha256": sha256(preprocessor),
        "prepared_split_sha256": {
            "source_train": split_sha256(source_train),
            "source_val": split_sha256(source_val),
            "target_val": split_sha256(target_val),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Saved {method} result: {output}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direction", required=True, choices=["unsw_to_cicids", "cicids_to_unsw"])
    parser.add_argument("--method", required=True, choices=["logistic_regression", "xgboost"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-rows", type=int, default=250000)
    args = parser.parse_args()
    run(args.direction, args.method, args.seed, args.max_train_rows)


if __name__ == "__main__":
    main()
