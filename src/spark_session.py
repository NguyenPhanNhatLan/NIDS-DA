"""Spark session defaults for local development; respect spark-submit settings."""
from pyspark import SparkConf
from pyspark.sql import SparkSession


def get_spark() -> SparkSession:
    config = SparkConf()
    defaults = {
        'spark.master': 'local[2]',
        'spark.app.name': 'KLTN-proposal-v2-data',
        'spark.sql.shuffle.partitions': '16',
        'spark.sql.files.maxPartitionBytes': '33554432',
        'spark.sql.parquet.columnarReaderBatchSize': '1024',
        # Dirty string numerics become null and are counted by data profiling.
        'spark.sql.ansi.enabled': 'false',
    }
    for key, value in defaults.items():
        if not config.contains(key):
            config.set(key, value)
    spark = SparkSession.builder.config(conf=config).getOrCreate()
    print(f'Spark master: {spark.sparkContext.master}', flush=True)
    return spark
