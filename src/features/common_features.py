from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/common_features_v1.json"
COMMON_FEATURES = tuple(item["canonical_name"] for item in json.loads(DEFAULT_CONFIG.read_text()))


def load_common_feature_config(path=DEFAULT_CONFIG):
    config = json.loads(Path(path).read_text())
    required = {"canonical_name", "unsw", "cicids", "direction", "transform", "reason"}
    if not isinstance(config, list) or not config or any(not required <= item.keys() for item in config):
        raise ValueError("Common feature config requires a nonempty list with all six fields")
    names = [item["canonical_name"] for item in config]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate canonical feature name")
    return config


def _numeric(frame, item, domain):
    column = item[domain]
    if column not in frame.columns:
        raise ValueError(f"Missing configured {domain} source column: {column}")
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)
    values[~np.isfinite(values)] = np.nan
    rule = item["transform"]
    if rule == "cicids_us_to_seconds" and domain == "cicids":
        values /= 1_000_000
    elif rule == "unsw_ms_cicids_us_to_seconds":
        values /= 1_000 if domain == "unsw" else 1_000_000
    elif rule not in {"none", "cicids_us_to_seconds"}:
        raise ValueError(f"Unknown transform: {rule}")
    return values


def fit_common_feature_pipeline(train_df, domain, config=None):
    if domain not in {"unsw", "cicids"}:
        raise ValueError(f"Unknown domain: {domain}")
    config = load_common_feature_config() if config is None else config
    medians = {}
    for item in config:
        values = _numeric(train_df, item, domain)
        finite = values[np.isfinite(values)]
        medians[item["canonical_name"]] = float(np.median(finite)) if len(finite) else 0.0
    return {"domain": domain, "features": [item["canonical_name"] for item in config], "medians": medians}


def transform_common_features(df, domain, fitted_state=None, config=None):
    if domain not in {"unsw", "cicids"}:
        raise ValueError(f"Unknown domain: {domain}")
    config = load_common_feature_config() if config is None else config
    names = [item["canonical_name"] for item in config]
    if fitted_state is None:
        raise ValueError("A training-fitted state is required")
    if fitted_state["domain"] != domain or fitted_state["features"] != names:
        raise ValueError("Fitted state domain or feature order differs from config")
    if "label" not in df.columns:
        raise ValueError("Missing label column")
    result = {}
    for item in config:
        name = item["canonical_name"]
        values = _numeric(df, item, domain)
        values[~np.isfinite(values)] = fitted_state["medians"][name]
        result[name] = values.astype(np.float32)
    output = pd.DataFrame(result, index=df.index)
    output["label"] = df["label"].to_numpy()
    return output


def _input_files(directory):
    files = sorted(directory.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet files in {directory}")
    return files


def _read_training_columns(directory, columns):
    chunks = []
    for path in _input_files(directory):
        chunks.append(pq.read_table(path, columns=columns).to_pandas())
    return pd.concat(chunks, ignore_index=True)


def generate_common_feature_datasets(split_root=ROOT / "data/splits", output_root=ROOT / "data/features/common_v1", config_path=DEFAULT_CONFIG):
    config = load_common_feature_config(config_path)
    split_root, output_root = Path(split_root), Path(output_root)
    names = [item["canonical_name"] for item in config]
    for domain in ("unsw", "cicids"):
        columns = [item[domain] for item in config]
        for split in ("train", "val", "test"):
            for path in _input_files(split_root / f"{domain}_{split}"):
                missing = set(columns + ["label"]) - set(pq.read_schema(path).names)
                if missing:
                    raise ValueError(f"Missing configured columns in {path}: {sorted(missing)}")
        train = _read_training_columns(split_root / f"{domain}_train", columns)
        state = fit_common_feature_pipeline(train, domain, config)
        del train
        target = output_root / f"{domain}_state.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(state, indent=2) + "\n")
        for split in ("train", "val", "test"):
            destination = output_root / f"{domain}_{split}"
            destination.mkdir(parents=True, exist_ok=True)
            for source in _input_files(split_root / f"{domain}_{split}"):
                with pq.ParquetFile(source) as parquet:
                    for batch_index, batch in enumerate(parquet.iter_batches(batch_size=65536, columns=columns + ["label"])):
                        transformed = transform_common_features(batch.to_pandas(), domain, state, config)
                        matrix = transformed[names].to_numpy(dtype=np.float32, copy=True)
                        vectors = pa.array(matrix.tolist(), type=pa.list_(pa.float32(), len(names)))
                        table = pa.table({"features": vectors, "label": pa.array(transformed["label"].to_numpy())})
                        pq.write_table(table, destination / f"{source.stem}-{batch_index:04d}.parquet", compression="snappy")
    return names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", type=Path, default=ROOT / "data/splits")
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/features/common_v1")
    args = parser.parse_args()
    names = generate_common_feature_datasets(args.split_root, args.output_root)
    print(f"Generated both domains with {len(names)} features: {', '.join(names)}")


if __name__ == "__main__":
    main()
