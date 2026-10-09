"""Integration fixture: native typed Spark hash, Parquet replay and split membership."""
import tempfile
import json
import unittest
from pathlib import Path

from pyspark.sql import functions as F, types as T
from features.splitting import split_data
from evaluation.proposal_split_replay import split_bucket, run
from features.common_features import DEFAULT_CONFIG
from spark_session import get_spark


class SparkSplitReplayTests(unittest.TestCase):
    def test_native_hash_bucket_and_membership_survive_parquet(self):
        spark = get_spark()
        try:
            schema = T.StructType([
                T.StructField('id', T.IntegerType(), False),
                T.StructField('dur', T.DoubleType()),
                T.StructField('spkts', T.IntegerType()),
                T.StructField('dpkts', T.LongType()),
                T.StructField('sbytes', T.FloatType()),
                T.StructField('dbytes', T.DoubleType()),
            ])
            rows = [(0, None, None, None, None, None),
                    (1, float('nan'), -2147483648, -9223372036854775808, float('nan'), -1.),
                    (2, -0., 0, 0, -0., 0.), (3, 0., 0, 0, 0., -0.),
                    (4, -1.25, -1, 9223372036854775807, -1.5, float('inf')),
                    (5, float('-inf'), 2147483647, 9007199254740993, float('inf'), float('nan'))]
            frame = spark.createDataFrame(rows, schema)
            keys = ['dur', 'spkts', 'dpkts', 'sbytes', 'dbytes']
            def values(x):
                return {r['id']: (r['hash'], r['bucket']) for r in x.select(
                    'id', F.xxhash64(*keys, F.lit(42)).alias('hash'), split_bucket(x, keys).alias('bucket')).collect()}
            expected = values(frame)
            self.assertEqual(expected[2], expected[3])
            with tempfile.TemporaryDirectory() as directory:
                frame.write.parquet(str(Path(directory) / 'fixture'))
                self.assertEqual(values(spark.read.parquet(str(Path(directory) / 'fixture'))), expected)
            parts = split_data(frame, 'unsw', 42)
            membership = {}
            for name, part in zip(('train', 'val', 'test'), parts):
                for row in part.select('id').collect():
                    self.assertNotIn(row['id'], membership)
                    membership[row['id']] = name
            for identity, (_, bucket) in expected.items():
                self.assertEqual(membership[identity], 'train' if bucket < 70 else 'val' if bucket < 85 else 'test')
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                config = json.loads(DEFAULT_CONFIG.read_text())
                for domain in ('unsw', 'cicids'):
                    mapped = frame.select(*[F.col(key).alias(item[domain]) for key, item in zip(keys, config)])
                    for name, part in zip(('train', 'val', 'test'), split_data(mapped, domain, 42)):
                        part.write.parquet(str(root / 'splits' / f'{domain}_{name}'))
                report = run(root / 'replay.json', root)
                self.assertTrue(report['quality_gate_passed'])
                self.assertEqual(report['unsw']['rows_replayed'], len(rows))
                self.assertEqual(report['cicids']['rows_replayed'], len(rows))
        finally:
            spark.stop()


if __name__ == '__main__':
    unittest.main()
