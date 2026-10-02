from pathlib import Path

import joblib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from models.proposal_pipeline import (
    proposal_processor,
)

ROOT = Path(__file__).resolve().parents[2]

RAW_ROOT = ROOT / "data/features/common_raw"

OUTPUT_ROOT = ROOT / "data/features/proposal_v1"


def load_dataset(path):
    files = sorted(Path(path).glob("*.parquet"))

    if not files:
        raise FileNotFoundError(path)

    xs = []
    ys = []

    for file in files:

        table = pq.read_table(
            file,
            columns=[
                "features",
                "label",
            ],
        )

        x = np.asarray(
            table["features"].to_pylist(),
            dtype=np.float64,
        )

        y = np.asarray(
            table["label"].to_pylist(),
            dtype=np.int64,
        )

        xs.append(x)
        ys.append(y)

    return (
        np.concatenate(xs),
        np.concatenate(ys),
    )


def direction_domains(direction):

    if direction == "unsw_to_cicids":
        return "unsw", "cicids"

    if direction == "cicids_to_unsw":
        return "cicids", "unsw"

    raise ValueError(direction)


def fit_direction_processor(direction):

    source, _ = direction_domains(direction)

    x_train, _ = load_dataset(RAW_ROOT / f"{source}_train")

    processor = proposal_processor()

    processor.fit(x_train)

    output_dir = ROOT / "models/proposal_v1" / direction

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = output_dir / "preprocessor.joblib"

    joblib.dump(
        processor,
        path,
    )

    print(f"Saved processor: {path}")

    return processor


def save_dataset(
    destination,
    x,
    y,
):
    destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    vectors = pa.array(
        x.astype(np.float32).tolist(),
        type=pa.list_(
            pa.float32(),
            x.shape[1],
        ),
    )

    table = pa.table(
        {
            "features": vectors,
            "label": pa.array(y.astype(np.int64)),
        }
    )

    pq.write_table(
        table,
        destination / "data.parquet",
        compression="snappy",
    )


def prepare_direction(direction):

    source, target = direction_domains(direction)

    processor = fit_direction_processor(direction)

    output_base = OUTPUT_ROOT / direction

    datasets = [
        (source, "train"),
        (source, "val"),
        (source, "test"),
        (target, "train"),
        (target, "val"),
        (target, "test"),
    ]

    for domain, split in datasets:

        x, y = load_dataset(RAW_ROOT / f"{domain}_{split}")

        # IMPORTANT:
        # transform only.
        # NEVER fit target.
        x = processor.transform(x)

        if not np.isfinite(x).all():
            raise ValueError(f"NaN/Inf after preprocessing: " f"{domain}_{split}")

        destination = output_base / f"{domain}_{split}"

        save_dataset(
            destination,
            x,
            y,
        )

        print(f"{direction} | " f"{domain}_{split} | " f"{x.shape}")


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--direction",
        required=True,
        choices=[
            "unsw_to_cicids",
            "cicids_to_unsw",
        ],
    )

    args = parser.parse_args()

    prepare_direction(args.direction)
