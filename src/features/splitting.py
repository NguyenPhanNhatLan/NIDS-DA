import argparse
from pathlib import Path

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from spark_session import get_spark

PROJECT_DIR = Path(__file__).resolve().parents[2]
SEED = 42


def split_data(df: DataFrame, seed: int = SEED):
    """Chia cố định 70/15/15 theo features; bản ghi trùng không qua hai tập."""
    feature_columns = [name for name in df.columns if name != "label"]
    if not feature_columns:
        raise ValueError("Không có features để tạo split cố định.")

    # Hash chỉ dùng đầu vào, không dùng nhãn. Cùng features luôn vào cùng tập.
    bucket = F.pmod(
        F.xxhash64(*[F.col(name) for name in feature_columns], F.lit(seed)),
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

    source_path = PROJECT_DIR / "data" / "processed" / f"clean_{args.dataset}.parquet"
    split_dir = PROJECT_DIR / "data" / "splits"
    paths = [split_dir / f"{args.dataset}_{name}" for name in ("train", "val", "test")]
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            f"Split đã có: {existing}. Không tự ghi đè để giữ cùng một seed {SEED}."
        )

    spark = get_spark()
    try:
        data = spark.read.parquet(str(source_path))
        if args.dataset == "cicids":
            data = data.drop("label").withColumnRenamed("binary_label", "label")

        if "label" not in data.columns:
            raise ValueError("Dữ liệu thiếu cột label.")
        label = F.col("label").cast("double")
        if data.filter(label.isNull() | ~label.isin(0, 1)).limit(1).count():
            raise ValueError("label phải là 0/1, không được null.")
        data = data.withColumn("label", F.col("label").cast("int"))

        for name, part, path in zip(("train", "val", "test"), split_data(data), paths):
            part.write.mode("errorifexists").parquet(str(path))
            print(f"{args.dataset} {name}: {path}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
