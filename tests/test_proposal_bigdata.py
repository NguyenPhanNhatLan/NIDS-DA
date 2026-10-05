"""Real Spark integration tests on small raw CSV fixtures, without training."""

import csv
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pyspark.sql import functions as F

from bigdata.clean import clean
from bigdata.harmonize import harmonize
from bigdata.ingest import normalize_headers
from bigdata.pipeline import run
from bigdata.preprocess import fit_source_processor, transform
from bigdata.profile import profile
from bigdata.split import split
from features.common_features import COMMON_FEATURES, load_common_feature_config
from features.parquet_vectors import vector_matrix
from training.proposal_data import ParquetBatchStream
from evaluation.proposal_domain_shift import sample_features
from evaluation.proposal_negative_transfer import sample_labeled
from experiments.proposal_classical_baselines import read_batches
from spark_session import get_spark


class ArrowVectorTests(unittest.TestCase):
    def test_spark_lists_and_sliced_fixed_lists(self):
        values = np.arange(30, dtype=np.float32).reshape(6, 5)
        fixed = pa.FixedSizeListArray.from_arrays(pa.array(values.ravel()), 5)
        variable = pa.array(values.tolist(), type=pa.list_(pa.float32()))
        for array in (fixed, variable):
            np.testing.assert_array_equal(vector_matrix(array.slice(2, 2)), values[2:4])
        with self.assertRaisesRegex(ValueError, "Expected 5 features"):
            vector_matrix(pa.array([[1.0] * 4], type=pa.list_(pa.float32())))
        with self.assertRaisesRegex(ValueError, "Null"):
            vector_matrix(pa.array([[1.0, None, 1, 1, 1]], type=pa.list_(pa.float32())))


class SparkDataLayerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = get_spark()
        cls.spark.sparkContext.setLogLevel("ERROR")
        cls.spark.conf.set("spark.sql.shuffle.partitions", "2")

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def raw(self, domain, rows):
        names = [item[domain] for item in load_common_feature_config()]
        return self.spark.createDataFrame(rows, names + ["label", "flow_id"])

    def test_clean_deduplicates_full_records_and_nulls_dirty_values(self):
        frame = self.raw(
            "unsw",
            [
                ("bad", "1", "-2", "3.5", "4", "1", "a"),
                ("bad", "1", "-2", "3.5", "4", "1", "a"),
                ("bad", "1", "-2", "3.5", "4", "1", "b"),
            ],
        )
        result = clean(frame, "unsw")
        self.assertEqual(
            result.count(), 2
        )
        row = result.first()
        self.assertIsNone(row.dur)
        self.assertIsNone(row.dpkts)
        self.assertIsNone(row.sbytes)
        self.assertEqual(row.label, 1)
        report = profile(result, "unsw")
        self.assertEqual(report["features"]["dur"]["missing"], 2)
        with self.assertRaisesRegex(ValueError, "Invalid unsw raw labels"):
            clean(self.raw("unsw", [("1", "1", "1", "1", "1", "1.9", "a")]), "unsw")

    def test_headers_and_cicids_units(self):
        raw = self.raw(
            "cicids",
            [
                ("2000000", "2", "3", "4", "5", " BENIGN ", "a"),
                ("1000000", "2", "3", "4", "5", "DoS", "b"),
            ],
        )
        mapped = harmonize(clean(raw, "cicids"), "cicids")
        values = {(row.flow_duration, row.label) for row in mapped.collect()}
        self.assertEqual(values, {(2.0, 0), (1.0, 1)})
        with self.assertRaisesRegex(ValueError, "Missing required"):
            normalize_headers(raw.drop("total_fwd_packets"), "cicids")
        with self.assertRaisesRegex(ValueError, "Invalid cicids raw labels"):
            clean(self.raw("cicids", [("1", "1", "1", "1", "1", "0", "a")]), "cicids")

    def test_split_is_disjoint_and_independent_of_partitions(self):
        rows = [(str(i), "1", "2", "3", "4", str(i % 2), str(i)) for i in range(150)]
        frame = clean(self.raw("unsw", rows), "unsw")
        sets = [
            {row.dur for row in part.select("dur").collect()}
            for part in split(frame, "unsw").values()
        ]
        again = [
            {row.dur for row in part.select("dur").collect()}
            for part in split(frame.repartition(3), "unsw").values()
        ]
        self.assertEqual(sets, again)
        self.assertEqual(len(set.union(*sets)), 150)
        self.assertTrue(all(sets))
        for i in range(3):
            for j in range(i + 1, 3):
                self.assertFalse(sets[i] & sets[j])

    def test_fit_requires_only_source_features_and_matches_spark_transform(self):
        # Source contains no labels; all-null feature and zero IQR remain supported.
        schema = "flow_duration double, fwd_packets double, bwd_packets double, fwd_bytes double, bwd_bytes double"
        source = self.spark.createDataFrame(
            [
                (1.0, 2.0, None, 3.0, 0.0),
                (3.0, 2.0, None, 5.0, 0.0),
                (None, 2.0, None, 7.0, 0.0),
            ],
            schema,
        )
        processor = fit_source_processor(source)
        self.assertEqual(processor.medians[2], 0.0)
        self.assertEqual(processor.scales[4], 1.0)
        target = self.spark.createDataFrame(
            [(1000.0, 2.0, None, 4.0, 0.0, 1)], schema + ", label long"
        )
        result = transform(target, processor).first()
        expected = processor.transform([[1000, 2, np.nan, 4, 0]])[0]
        np.testing.assert_allclose(result.features, expected, rtol=1e-6, atol=1e-6)
        self.assertLess(processor.medians[0], 1000)
        with self.assertRaisesRegex(ValueError, "relative_error"):
            fit_source_processor(source, 0)

    def test_raw_csv_to_parquet_to_all_batch_consumers(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            inputs = {}
            for domain in ("unsw", "cicids"):
                path = root / f"{domain}.csv"
                inputs[domain] = [path]
                columns = [item[domain] for item in load_common_feature_config()]
                header = [column.replace("_", " ").title() for column in columns]
                with path.open("w", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(header + ["Label", "Flow ID"])
                    for i in range(100):
                        duration = (i + 1) * (1000000 if domain == "cicids" else 1)
                        label = (
                            str(i % 2)
                            if domain == "unsw"
                            else ("BENIGN" if i % 2 == 0 else "DoS")
                        )
                        writer.writerow(
                            [duration, i + 1, i + 2, i + 3, i + 4, label, i]
                        )
            work, features, models = root / "work", root / "features", root / "models"
            manifest = run(self.spark, inputs, work, features, models)
            self.assertTrue((work / "manifest.json").is_file())
            self.assertEqual(manifest["data_revision"], "spark_data_v1")
            for direction in ("unsw_to_cicids", "cicids_to_unsw"):
                processor = joblib.load(models / direction / "preprocessor.joblib")
                self.assertEqual(len(processor.medians), 5)
                for domain in ("unsw", "cicids"):
                    for part in ("train", "val", "test"):
                        path = features / direction / f"{domain}_{part}"
                        batches = list(ParquetBatchStream(path, 7, False, 42, True))
                        self.assertTrue(batches)
                        self.assertTrue(all(x.shape[1] == 5 for x, _ in batches))
                        self.assertTrue(
                            all(np.isfinite(x.numpy()).all() for x, _ in batches)
                        )
            path = features / "unsw_to_cicids" / "cicids_val"
            sample_feature, total_rows = sample_features(path, 5, 42)
            self.assertEqual(sample_feature.shape, (5, 5))
            self.assertGreaterEqual(total_rows, len(sample_feature))
            sampled, labels = sample_labeled(path, 5, 42)
            self.assertEqual(sampled.shape, (5, 5))
            self.assertEqual(len(labels), 5)
            self.assertGreater(sum(len(x) for x, y in read_batches(path)), 0)
            with self.assertRaises(FileExistsError):
                run(self.spark, inputs, work, features, models)


if __name__ == "__main__":
    unittest.main()
