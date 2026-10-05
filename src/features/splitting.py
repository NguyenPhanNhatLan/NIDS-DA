import argparse
from pathlib import Path

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from spark_session import get_spark

PROJECT_DIR = Path(__file__).resolve().parents[2]
SEED = 42


SPLIT_KEY_COLUMNS = {
    "unsw": [
        "dur",
        "spkts",
        "dpkts",
        "sbytes",
        "dbytes",
    ],
    "cicids": [
        "flow_duration",
        "total_fwd_packets",
        "total_backward_packets",
        "total_length_of_fwd_packets",
        "total_length_of_bwd_packets",
    ],
}


def split_data(df: DataFrame, dataset: str, seed: int = SEED):
    key_columns = SPLIT_KEY_COLUMNS[dataset]

    missing = [c for c in key_columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing split-key columns for {dataset}: {missing}"
        )

    bucket = F.pmod(
        F.xxhash64(
            *[F.col(c) for c in key_columns],
            F.lit(seed),
        ),
        F.lit(100),
    )

    train_df = df.filter(bucket < 70)
    val_df = df.filter((bucket >= 70) & (bucket < 85))
    test_df = df.filter(bucket >= 85)

    return train_df, val_df, test_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["unsw", "cicids"])
    args = parser.parse_args()

    source_path = PROJECT_DIR / "data" / "processed" / f"{args.dataset}_common5.parquet"
    split_dir = PROJECT_DIR / "data" / "splits_v2"
    paths = [split_dir / f"{args.dataset}_{name}" for name in ("train", "val", "test")]
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            f"Split đã có: {existing}. Không tự ghi đè để giữ cùng một seed {SEED}."
        )

    spark = get_spark()
    try:
        data = spark.read.parquet(str(source_path))

        if "label" not in data.columns:
            raise ValueError("Dữ liệu thiếu cột label.")
        label = F.col("label").cast("double")
        if data.filter(label.isNull() | ~label.isin(0, 1)).limit(1).count():
            raise ValueError("label phải là 0/1, không được null.")
        data = data.withColumn("label", F.col("label").cast("int"))

        for name, part, path in zip(("train", "val", "test"), split_data(data, args.dataset), paths):
            part.write.mode("errorifexists").parquet(str(path))
            print(f"{args.dataset} {name}: {path}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
