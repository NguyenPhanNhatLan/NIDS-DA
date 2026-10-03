from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/common_features_v2.json"
COMMON_FEATURES = tuple(
    item["canonical_name"] for item in json.loads(DEFAULT_CONFIG.read_text())
)


def load_common_feature_config(path=DEFAULT_CONFIG):
    config = json.loads(Path(path).read_text())

    required = {
        "canonical_name",
        "unsw",
        "cicids",
        "direction",
        "transform",
        "reason",
    }

    if (
        not isinstance(config, list)
        or not config
        or any(not required <= item.keys() for item in config)
    ):
        raise ValueError("Invalid common feature config")

    names = [item["canonical_name"] for item in config]

    if len(names) != len(set(names)):
        raise ValueError("Duplicate canonical names")

    return config


def canonicalize_column(
    frame,
    item,
    domain,
):
    if domain not in {
        "unsw",
        "cicids",
    }:
        raise ValueError(f"Unknown domain: {domain}")

    source_column = item[domain]

    if source_column not in frame.columns:
        raise ValueError(f"Missing {domain} column: " f"{source_column}")

    values = pd.to_numeric(
        frame[source_column],
        errors="coerce",
    ).to_numpy(
        dtype=np.float64,
        na_value=np.nan,
        copy=True,
    )

    values[~np.isfinite(values)] = np.nan

    transform = item["transform"]

    if transform == "cicids_us_to_seconds":

        if domain == "cicids":
            values /= 1_000_000.0

    elif transform == "unsw_ms_cicids_us_to_seconds":

        if domain == "unsw":
            values /= 1_000.0

        else:
            values /= 1_000_000.0

    elif transform == "none":
        pass

    else:
        raise ValueError(f"Unknown transform: {transform}")

    return values.astype(np.float32)


def canonicalize_frame(
    frame,
    domain,
    config,
):
    if domain not in {"unsw", "cicids"}:
        raise ValueError(domain)

    result = {}

    for item in config:

        result[item["canonical_name"]] = canonicalize_column(
            frame,
            item,
            domain,
        )

    output = pd.DataFrame(result)

    if "label" not in frame.columns:
        raise ValueError("Missing label")

    output["label"] = (
        pd.to_numeric(
            frame["label"],
            errors="raise",
        )
        .astype(np.int64)
        .to_numpy()
    )

    return output


def _input_files(directory):
    files = sorted(directory.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet files in {directory}")
    return files


def generate_raw_common_feature_datasets(
    split_root=ROOT / "data/splits",
    output_root=ROOT / "data/features/common_raw",
    config_path=DEFAULT_CONFIG,
):
    config = load_common_feature_config(config_path)

    split_root = Path(split_root)
    output_root = Path(output_root)

    names = [item["canonical_name"] for item in config]

    for domain in ("unsw", "cicids"):

        source_columns = [item[domain] for item in config]

        for split in ("train", "val", "test"):

            source_dir = split_root / f"{domain}_{split}"

            for path in _input_files(source_dir):

                available = set(pq.read_schema(path).names)

                required = set(source_columns + ["label"])

                missing = required - available

                if missing:
                    raise ValueError(
                        f"Missing configured columns " f"in {path}: {sorted(missing)}"
                    )

        for split in ("train", "val", "test"):

            source_dir = split_root / f"{domain}_{split}"

            destination = output_root / f"{domain}_{split}"

            if destination.exists():
                raise FileExistsError(f"Output already exists: " f"{destination}")

            destination.mkdir(
                parents=True,
                exist_ok=False,
            )

            for file_index, source in enumerate(_input_files(source_dir)):

                with pq.ParquetFile(source) as parquet:

                    for batch_index, batch in enumerate(
                        parquet.iter_batches(
                            batch_size=65536,
                            columns=source_columns + ["label"],
                        )
                    ):

                        transformed = canonicalize_frame(
                            batch.to_pandas(),
                            domain,
                            config,
                        )

                        matrix = transformed[names].to_numpy(
                            dtype=np.float32,
                            copy=True,
                        )

                        vectors = pa.array(
                            matrix.tolist(),
                            type=pa.list_(
                                pa.float32(),
                                len(names),
                            ),
                        )

                        labels = transformed["label"].to_numpy(dtype=np.int64)

                        if not np.isin(
                            labels,
                            [0, 1],
                        ).all():
                            raise ValueError(f"Invalid binary labels " f"in {source}")

                        table = pa.table(
                            {
                                "features": vectors,
                                "label": pa.array(labels),
                            }
                        )

                        output_file = destination / (
                            f"part-"
                            f"{file_index:04d}-"
                            f"{batch_index:04d}"
                            ".parquet"
                        )

                        pq.write_table(
                            table,
                            output_file,
                            compression="snappy",
                        )

    return names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", type=Path, default=ROOT / "data/splits")
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "data/features/common_raw2"
    )
    args = parser.parse_args()
    names = generate_raw_common_feature_datasets(args.split_root, args.output_root)
    print(f"Generated both domains with {len(names)} features: {', '.join(names)}")


if __name__ == "__main__":
    main()
