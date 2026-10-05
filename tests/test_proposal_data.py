import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from features import proposal_preprocessing as prep
from training.proposal_data import iter_parquet_batches, split_sha256


def write_vectors(directory, values, labels=None):
    directory.mkdir(parents=True)
    matrix = np.asarray(values, dtype=np.float32)
    vectors = pa.FixedSizeListArray.from_arrays(pa.array(matrix.ravel()), 5)
    columns = {"features": vectors}
    if labels is not None:
        columns["label"] = pa.array(labels, type=pa.int64())
    pq.write_table(pa.table(columns), directory / "data.parquet")


class ProposalDataTests(unittest.TestCase):
    def test_batch_order_and_features_only_target(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source"
            target = Path(temp) / "target"
            rows = np.arange(27 * 5, dtype=np.float32).reshape(27, 5)
            labels = np.arange(27) % 2
            write_vectors(source, rows, labels)
            write_vectors(target, rows)
            initial_hash = split_sha256(source)
            original = (source / "data.parquet").read_bytes()
            with (source / "data.parquet").open("ab") as stream:
                stream.write(b"changed")
            self.assertNotEqual(initial_hash, split_sha256(source))
            (source / "data.parquet").write_bytes(original)
            first = list(iter_parquet_batches(source, 8, True, 42, True))
            second = list(iter_parquet_batches(source, 8, True, 42, True))
            self.assertTrue(all(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
                                for a, b in zip(first, second)))
            self.assertEqual(sum(len(x) for x, _ in first), 27)
            ordered = list(iter_parquet_batches(source, 8, False, 42, True))
            self.assertTrue(np.array_equal(np.concatenate([x.numpy() for x, _ in ordered]), rows))
            target_batches = list(iter_parquet_batches(target, 8, False, 42, False))
            self.assertEqual(sum(len(x) for x in target_batches), 27)

    def test_preprocessing_fits_source_only_and_streams_all_splits(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = root / "common_raw"
            output = root / "proposal_v2"
            source = np.zeros((3, 5), dtype=np.float32)
            source[:, 0] = [1, 3, np.nan]
            target = np.zeros((2, 5), dtype=np.float32)
            target[:, 0] = [1000, np.nan]
            for domain, values in (("unsw", source), ("cicids", target)):
                for split in ("train", "val", "test"):
                    write_vectors(raw / f"{domain}_{split}", values, np.arange(len(values)) % 2)
            with patch.object(prep, "ROOT", root), patch.object(prep, "RAW_ROOT", raw), \
                 patch.object(prep, "OUTPUT_ROOT", output):
                prep.prepare_direction("unsw_to_cicids")
                processor = joblib.load(root / "models/proposal_v2/unsw_to_cicids/preprocessor.joblib")
                self.assertEqual(processor.named_steps["imputer"].statistics_[0], 2.0)
                target_output = pq.read_table(output / "unsw_to_cicids/cicids_val/data.parquet")
                self.assertEqual(target_output.num_rows, 2)
                self.assertEqual(target_output.schema.field("features").type.list_size, 5)
                self.assertEqual(target_output["label"].to_pylist(), [0, 1])
                self.assertTrue(np.isfinite(target_output["features"].combine_chunks().values.to_numpy()).all())
                with self.assertRaises(FileExistsError):
                    prep.prepare_direction("unsw_to_cicids")


if __name__ == "__main__":
    unittest.main()
