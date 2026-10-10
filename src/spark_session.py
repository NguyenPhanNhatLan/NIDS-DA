"""Spark session defaults for local development; respect spark-submit settings."""
import os
from pathlib import Path

import pyspark
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession


def get_spark() -> SparkSession:
    # Plain Python must launch the JVM bundled with this PySpark installation.
    # spark-submit supplies its own gateway; keep that environment untouched.
    if SparkContext._gateway is None and 'PYSPARK_GATEWAY_PORT' not in os.environ:
        bundled_home = Path(pyspark.__file__).resolve().parent
        if not (bundled_home / 'bin' / 'spark-submit').is_file() or not (
            bundled_home / 'jars'
        ).is_dir():
            raise RuntimeError(
                f'Bundled Spark runtime is missing from {bundled_home}; '
                'reinstall the project bigdata dependencies in the active virtualenv.'
            )
        os.environ['SPARK_HOME'] = str(bundled_home)

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
