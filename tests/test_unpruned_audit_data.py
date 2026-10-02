"""Spark integration tests: run with spark-submit to exercise actual join semantics."""
import json
from pathlib import Path
import tempfile
import unittest

from features.unpruned_audit_data import attach_boundaries, normalized_names, notebook_removals, ROOT


class NotebookAuditTests(unittest.TestCase):
    def test_column_normalization(self):
        self.assertEqual(normalized_names([' A/B ', 'a_b', 'Fwd Header Length.1']),
                         ['a_b', 'a_b_1', 'fwd_header_length_1'])

    def test_exact_recorded_removals(self):
        import csv
        import pyarrow.parquet as pq
        with (ROOT / 'data/raw/CICIDS2017.csv').open() as f:
            original = normalized_names(next(csv.reader(f)))
        existing = pq.ParquetDataset(ROOT / 'data/splits/cicids_train').schema.names
        audit = notebook_removals(ROOT / 'notebooks/cicids.ipynb', original, existing)
        self.assertEqual(len(audit['constant_removed']), 4)
        self.assertEqual(len(audit['near_constant_removed']), 4)
        self.assertEqual(len(audit['correlation_removed']), 24)
        self.assertEqual(len(audit['constant_requested_but_not_removed']), 4)
        self.assertIn('total_backward_packets', audit['correlation_removed'])
        self.assertEqual(audit['other_manual_removed'], [])


class BoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from pyspark.sql import SparkSession
        cls.spark = (SparkSession.builder.master('local[2]').appName('boundary-tests')
                     .config('spark.sql.shuffle.partitions', '2').getOrCreate())
        cls.spark.sparkContext.setLogLevel('ERROR')

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def test_exact_join_not_position_and_multiplicity(self):
        raw = self.spark.createDataFrame([(2., 20.), (1., 11.), (1., 12.)], ['kept','recovered'])
        old = {'train': self.spark.createDataFrame([(1.,), (1.,)], ['kept']),
               'test': self.spark.createDataFrame([(2.,)], ['kept'])}
        joined, quarantine, cache, report = attach_boundaries(raw, old, ['kept'])
        self.assertEqual({(r.recovered, r._split) for r in joined.collect()},
                         {(11.,'train'), (12.,'train'), (20.,'test')})
        self.assertEqual(report['usable_counts'], {'train':2,'test':1})
        self.assertEqual(quarantine.count(), 0)
        cache.unpersist()

    def test_ambiguity_fails_closed_or_quarantines(self):
        raw = self.spark.createDataFrame([(1., 11.), (1., 12.), (2., 20.)], ['kept','recovered'])
        old = {'train': self.spark.createDataFrame([(1.,), (2.,)], ['kept']),
               'test': self.spark.createDataFrame([(1.,)], ['kept'])}
        with self.assertRaisesRegex(ValueError, 'Ambiguous provenance'):
            attach_boundaries(raw, old, ['kept'])
        joined, quarantine, cache, report = attach_boundaries(raw, old, ['kept'], True)
        self.assertEqual(joined.count(), 1)
        self.assertEqual(report['ambiguous_rows'], 2)
        self.assertEqual({r.recovered for r in quarantine.collect()}, {11.,12.})
        self.assertNotIn('_split', quarantine.columns)
        cache.unpersist()

    def test_missing_or_extra_raw_record_rejected(self):
        raw = self.spark.createDataFrame([(1., 11.), (1., 12.)], ['kept','recovered'])
        old = {'train': self.spark.createDataFrame([(1.,)], ['kept'])}
        with self.assertRaisesRegex(ValueError, 'multisets differ'):
            attach_boundaries(raw, old, ['kept'])

    def test_null_keys_match_without_approximation(self):
        raw = self.spark.createDataFrame([(None, 11.), (2., 20.)], 'kept double, recovered double')
        old = {'train': self.spark.createDataFrame([(None,), (2.,)], 'kept double')}
        joined, _, cache, _ = attach_boundaries(raw, old, ['kept'])
        self.assertEqual(joined.count(), 2)
        cache.unpersist()


if __name__ == '__main__':
    unittest.main(verbosity=2)
