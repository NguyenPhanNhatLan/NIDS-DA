"""Đánh giá trực tiếp checkpoint UNSW trên CICIDS, không train model đích."""
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from evaluation.baseline import evaluate
from models.baseline import BaselineMLP
from training.baseline import make_loader


PROJECT_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_DIR / "models/unsw_mlp.pt"
TARGET_PATH = PROJECT_DIR / "data/processed/clean_cicids.parquet"
LABEL_COLUMN = "binary_label"
BATCH_SIZE = 256


def log_features(data, feature_columns):
    from pyspark.sql import functions as F

    selected_columns = []
    for name in feature_columns:
        value = F.col(name).cast("double")
        is_invalid = (
            value.isNull()
            | F.isnan(value)
            | (F.abs(value) == float("inf"))
        )

        # Signed log1p: số dương -> log1p(x), số âm -> -log1p(-x).
        log_value = F.signum(value) * F.log1p(F.abs(value))
        clean_value = F.when(is_invalid, F.lit(None)).otherwise(log_value)
        selected_columns.append(clean_value.alias(name))

    if "label" in data.columns:
        selected_columns.append(F.col("label"))

    return data.select(*selected_columns)


def build_cicids_pipeline(feature_columns):
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import Imputer, VectorAssembler, StandardScaler

    imputed_columns = [name + "__imputed" for name in feature_columns]

    imputer = Imputer(
        inputCols=feature_columns,
        outputCols=imputed_columns,
        strategy="median",
    )
    assembler = VectorAssembler(
        inputCols=imputed_columns,
        outputCol="raw_features",
    )
    scaler = StandardScaler(
        inputCol="raw_features",
        outputCol="features",
        withMean=True,
        withStd=True,
    )
    return Pipeline(stages=[imputer, assembler, scaler])


def prepare_cicids(spark, temp_dir, model_dim):
    from pyspark.ml.functions import vector_to_array
    from pyspark.sql import functions as F
    from pyspark.sql.types import NumericType

    print("[1/4] Đọc CICIDS và chọn các cột đầu vào...", flush=True)
    cicids = spark.read.parquet(str(TARGET_PATH))
    if LABEL_COLUMN not in cicids.columns:
        raise ValueError(f"CICIDS thiếu cột nhãn: {LABEL_COLUMN}")

    label_columns = {"label", "binary_label", "label_clean", "attack_cat"}
    label_columns.add(LABEL_COLUMN.lower())
    feature_columns = []
    for field in cicids.schema.fields:
        if field.name.lower() in label_columns:
            continue
        if not isinstance(field.dataType, NumericType):
            raise ValueError(f"Feature {field.name} chưa phải dạng số.")
        feature_columns.append(field.name)

    if not feature_columns:
        raise ValueError("CICIDS không có features dạng số.")
    target_dim = len(feature_columns)
    print(f"Số features CICIDS: {target_dim} | Model UNSW: {model_dim}", flush=True)
    if target_dim != model_dim:
        raise ValueError(
            f"CICIDS có {target_dim} features nhưng checkpoint UNSW cần {model_dim}. "
            "Không thể đánh giá trực tiếp checkpoint này với đầu vào hiện tại."
        )

    print("[2/4] Chia CICIDS thành phần fit preprocessing và test...", flush=True)
    # Hash chia thành 10 nhóm: 0..7 để fit preprocessing, 8..9 để test.
    # Các dòng có cùng features ở cùng nhóm; không dùng nhãn để chia.
    split_group = F.pmod(F.xxhash64(*feature_columns), F.lit(10))
    preprocessing_data = cicids.filter(split_group < 8).select(*feature_columns)
    test_data = cicids.filter(split_group >= 8)

    if preprocessing_data.limit(1).count() == 0:
        raise ValueError("Phần fit preprocessing không có dữ liệu.")
    if test_data.limit(1).count() == 0:
        raise ValueError("CICIDS test không có dữ liệu.")

    test_label = F.col(LABEL_COLUMN).cast("double")
    invalid_label = test_label.isNull() | ~test_label.isin(0, 1)
    if test_data.filter(invalid_label).limit(1).count() > 0:
        raise ValueError("Nhãn CICIDS test phải là 0 hoặc 1.")
    test_data = test_data.select(
        *feature_columns,
        test_label.cast("int").alias("label"),
    )

    print("[3/4] Log features, fit imputer và scaler trên phần preprocessing...", flush=True)
    preprocessing_data = log_features(preprocessing_data, feature_columns)
    test_data = log_features(test_data, feature_columns)

    pipeline = build_cicids_pipeline(feature_columns)
    processor = pipeline.fit(preprocessing_data)
    test_features = processor.transform(test_data)

    print("[4/4] Ghi features tạm để PyTorch đọc từng batch...", flush=True)
    test_path = temp_dir / "test"

    test_output = test_features.select(
        vector_to_array("features", dtype="float32").alias("features"),
        "label",
    )
    test_output.write.mode("errorifexists").parquet(str(test_path))

    return test_path


def main():
    torch.manual_seed(42)
    if BATCH_SIZE < 1:
        raise ValueError("BATCH_SIZE phải >= 1.")
    for path in (MODEL_PATH, TARGET_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy: {path}")

    # 1. Load model nguồn đã train; không thay đổi checkpoint UNSW.
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    source_dim = int(checkpoint["input_dim"])
    source_model = BaselineMLP(source_dim)
    source_model.load_state_dict(checkpoint["model_state_dict"])
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Thiết bị: {device} | Số features UNSW: {source_dim}", flush=True)
    source_model.to(device)

    # 2. Tạo features tạm. Dừng Spark trước khi dự đoán để giảm RAM.
    from spark_session import get_spark

    with TemporaryDirectory(prefix="cicids_baseline_") as directory:
        spark = get_spark()
        try:
            test_path = prepare_cicids(spark, Path(directory), source_dim)
        finally:
            spark.stop()

        test_loader = make_loader(
            test_path, input_dim=source_dim, batch_size=BATCH_SIZE, training=False,
        )

        # Chỉ inference: không optimizer, không MMD, không encoder đích.
        # Preprocessing fit trên CICIDS nên không phải source-only preprocessing.
        print("Đánh giá trực tiếp; cùng số chiều chưa đảm bảo cùng ý nghĩa features.")
        evaluate(source_model, test_loader, "CICIDS - model UNSW, không adaptation")


if __name__ == "__main__":
    main()
