from pyspark import SparkConf
from pyspark.sql import SparkSession


def get_spark() -> SparkSession:
    config = SparkConf()

    # Mặc định dùng 2 task đồng thời trên máy cá nhân.
    # Nếu đã truyền --master bằng spark-submit thì giữ cấu hình đó.
    if not config.contains("spark.master"):
        config.setMaster("local[2]")

    spark = (
        SparkSession.builder
        .config(conf=config)
        .appName("UNSW-MLP")
        .config("spark.sql.shuffle.partitions", "16")
        # Giảm kích thước partition khi đọc file: 32 MB.
        .config("spark.sql.files.maxPartitionBytes", "33554432")
        # Giảm số dòng mỗi batch của bộ đọc Parquet.
        .config("spark.sql.parquet.columnarReaderBatchSize", "1024")
        .getOrCreate()
    )

    # Heap phải đặt lúc khởi chạy: spark-submit --driver-memory 6g ...
    print(f"Spark master: {spark.sparkContext.master}", flush=True)
    return spark
