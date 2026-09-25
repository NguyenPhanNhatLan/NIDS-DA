import argparse
from pathlib import Path

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from spark_session import get_spark

PROJECT_DIR = Path(__file__).resolve().parents[2]
SEED = 42


def split_data(df: DataFrame, seed: int = SEED):
    return df.randomSplit([0.70, 0.15, 0.15], seed=seed)


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
