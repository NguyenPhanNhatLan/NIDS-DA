import json
from pathlib import Path

from pyspark.sql import functions as F
from pyspark.sql.types import NumericType
from pyspark.sql import DataFrame
from pyspark.ml import Pipeline
from pyspark.ml.functions import vector_to_array
from pyspark.ml.feature import (
    Imputer,
    SQLTransformer,
    StringIndexer,
    OneHotEncoder,
    VectorAssembler,
    StandardScaler,
)
from spark_session import get_spark

UNSW_LOG_CANDIDATES = [
    "dur",
    "sbytes",
    "dbytes",
    "sloss",
    "dloss",
    "sload",
    "spkts",
    "dpkts",
    "smeansz",
    "res_bdy_len",
    "sjit",
    "djit",
    "sintpkt",
    "dintpkt",
    "tcprtt",
    "ct_srv_src",
    "ct_srv_dst",
    "ct_dst_ltm",
    "ct_src__ltm",
    "ct_src_dport_ltm",
    "ct_dst_sport_ltm",
    "ct_dst_src_ltm",
]

PROJECT_DIR = Path(__file__).resolve().parents[2]
FEATURE_DIR = PROJECT_DIR / "data" / "features"

UNSW_CAT_COLS = ["proto", "service", "state"]

def prepare_dataset(dataset):
    spark = get_spark()

    try:
        split_dir = (
            PROJECT_DIR
            / "data"
            / "splits"
        )

        train_df = spark.read.parquet(
            str(
                split_dir
                / f"{dataset}_train"
            )
        )

        val_df = spark.read.parquet(
            str(
                split_dir
                / f"{dataset}_val"
            )
        )

        test_df = spark.read.parquet(
            str(
                split_dir
                / f"{dataset}_test"
            )
        )

        # -----------------------------
        # Validate labels
        # -----------------------------

        rows = (
            train_df
            .groupBy("label")
            .count()
            .collect()
        )

        counts = {}

        for row in rows:
            label = row["label"]

            if label not in (0, 1):
                raise ValueError(
                    f"Invalid label: {label}"
                )

            counts[int(label)] = int(
                row["count"]
            )

        if set(counts) != {0, 1}:
            raise ValueError(
                f"Need both classes: {counts}"
            )

        # -----------------------------
        # Preprocessing
        # -----------------------------

        if dataset == "unsw":
            (
                train_features,
                val_features,
                test_features,
            ) = unsw_featuring(
                train_df,
                val_df,
                test_df,
            )

        elif dataset == "cicids":
            (
                train_features,
                val_features,
                test_features,
            ) = cicids_featuring(
                train_df,
                val_df,
                test_df,
            )

        else:
            raise ValueError(dataset)

        first_row = train_features.first()

        input_dim = len(
            first_row["features"]
        )

        # -----------------------------
        # Save features
        # -----------------------------

        FEATURE_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        export_features(
            train_features,
            str(
                FEATURE_DIR
                / f"{dataset}_train"
            ),
        )

        export_features(
            val_features,
            str(
                FEATURE_DIR
                / f"{dataset}_val"
            ),
        )

        export_features(
            test_features,
            str(
                FEATURE_DIR
                / f"{dataset}_test"
            ),
        )

        metadata = {
            "dataset": dataset,
            "class_counts": counts,
            "input_dim": input_dim,
        }

        # Ghi vào file tạm rồi đổi tên sau khi JSON đã ghi xong.
        metadata_path = FEATURE_DIR / f"{dataset}_metadata.json"
        temporary_path = FEATURE_DIR / f"{dataset}_metadata.json.tmp"
        try:
            with temporary_path.open("w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
            temporary_path.replace(metadata_path)
        finally:
            temporary_path.unlink(missing_ok=True)

        print(
            f"{dataset}: "
            f"input_dim={input_dim}"
        )

    finally:
        spark.stop()
        
        
def featuring(numeric_cols, cat_cols, log_cols):
    stages = [
        Imputer(
            inputCols=numeric_cols,
            outputCols=[f"{col}__imp" for col in numeric_cols],
            strategy="median",
        )
    ]
    if log_cols:
        log_expressions = [f"LOG1P(`{col}__imp`) AS `{col}__log`" for col in log_cols]

        statement = "SELECT *, " + ", ".join(log_expressions) + " FROM __THIS__"

        log_transformer = SQLTransformer(statement=statement)

        stages.append(log_transformer)

    numeric_inputs = []
    for col in numeric_cols:
        if col in log_cols:
            numeric_inputs.append(f"{col}__log")
        else:
            numeric_inputs.append(f"{col}__imp")

    numeric_assembler = VectorAssembler(
        inputCols=numeric_inputs, outputCol="numeric_vector"
    )

    scaler = StandardScaler(
        inputCol="numeric_vector",
        outputCol="numeric_scaled",
        withMean=True,
        withStd=True,
    )

    stages.extend([numeric_assembler, scaler])

    encoded_cols = []

    for col in cat_cols:
        indexed_col = f"{col}_idx"
        encoded_col = f"{col}_ohe"

        indexer = StringIndexer(
            inputCol=col, outputCol=indexed_col, handleInvalid="keep"
        )

        encoder = OneHotEncoder(
            inputCol=indexed_col,
            outputCol=encoded_col,
            handleInvalid="keep",
            dropLast=False,
        )

        stages.extend([indexer, encoder])

        encoded_cols.append(encoded_col)

    final_assembler = VectorAssembler(
        inputCols=["numeric_scaled"] + encoded_cols,
        outputCol="features",
        handleInvalid="error",
    )

    stages.append(final_assembler)

    return Pipeline(stages=stages)


def export_features(df, output_path):

    output_df = df.select(
        vector_to_array("features", dtype="float32").alias("features"),

        F.col("label").cast("int").alias("label")
    )

    output_df.write.mode("overwrite").parquet(output_path)


def unsw_featuring(train_df: DataFrame, val_df: DataFrame, test_df: DataFrame):
    excluded = {"label", "binary_label", "label_clean", "attack_cat", "id"}

    numeric_cols = [
        field.name
        for field in train_df.schema.fields
        if isinstance(field.dataType, NumericType)
        and field.name.lower() not in excluded
        and field.name not in UNSW_CAT_COLS
    ]

    log_cols = [c for c in UNSW_LOG_CANDIDATES if c in numeric_cols]

    pipeline = featuring(
        numeric_cols=numeric_cols, cat_cols=UNSW_CAT_COLS, log_cols=log_cols
    )

    # Chỉ fit trên train; dùng cùng phép biến đổi cho validation và test.
    processor = pipeline.fit(train_df)

    train_features = processor.transform(train_df).select("features", "label")
    val_features = processor.transform(val_df).select("features", "label")
    test_features = processor.transform(test_df).select("features", "label")

    processor.write().overwrite().save(str(PROJECT_DIR / "models" / "unsw_feature_pipeline"))
    return train_features, val_features, test_features

def cicids_featuring(
    train_df,
    val_df,
    test_df,
):
    excluded = {
        "label",
        "binary_label",
        "label_clean",
        "attack_cat",
        "id",
    }

    numeric_cols = [
        field.name
        for field in train_df.schema.fields
        if isinstance(field.dataType, NumericType)
        and field.name.lower() not in excluded
    ]

    if not numeric_cols:
        raise ValueError(
            "CICIDS không có numeric features."
        )

    log_cols = ['total_length_of_fwd_packets', 'total_fwd_packets', 'active_min', 'active_std', 'active_mean', 'active_max', 'bwd_packets_s', 'fwd_packet_length_min', 'down_up_ratio', 'min_packet_length', 'idle_std', 'bwd_iat_min', 'fwd_packet_length_max', 'fwd_packet_length_mean', 'bwd_iat_mean', 'fwd_iat_mean', 'bwd_iat_std', 'packet_length_variance', 'bwd_packet_length_min', 'bwd_iat_max', 'flow_iat_std', 'fwd_iat_std', 'bwd_packet_length_max', 'bwd_iat_total', 'max_packet_length', 'packet_length_mean']

    pipeline = featuring(
        numeric_cols=numeric_cols,
        cat_cols=[],
        log_cols=log_cols,
    )

    # Fit CHỈ trên CICIDS train.
    processor = pipeline.fit(train_df)

    train_features = (
        processor
        .transform(train_df)
        .select("features", "label")
    )

    val_features = (
        processor
        .transform(val_df)
        .select("features", "label")
    )

    test_features = (
        processor
        .transform(test_df)
        .select("features", "label")
    )

    processor.write().overwrite().save(
        str(
            PROJECT_DIR
            / "models"
            / "cicids_feature_pipeline"
        )
    )

    return (
        train_features,
        val_features,
        test_features,
    )

def main():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
        choices=["unsw", "cicids"],
    )

    args = parser.parse_args()

    prepare_dataset(
        args.dataset
    )


if __name__ == "__main__":
    main()
