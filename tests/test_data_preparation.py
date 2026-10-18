"""Regression checks for frozen splits and the Parquet training stream."""

import io
import json
import os
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from evaluation.proposal_data_audit import DOMAINS, SPLITS, inventory
from features import freeze_proposal_revision as freezing
from features.common_features import COMMON_FEATURES
from features.parquet_vectors import vector_matrix
from models.proposal_pipeline import proposal_processor
from training import data_revision
from training.proposal_data import ParquetBatchStream, iter_parquet_batches


def write_audited_inputs(root):
    """One train row rounds to the validation vector after float32 conversion."""
    (root / "configs").mkdir(parents=True)
    (root / "configs/common_features_v2.json").write_text("[]\n")
    source_manifest = root / "source_manifest.json"
    source_manifest.write_text(json.dumps({"raw_inputs": {}}))
    common_inventory = {}
    for domain in DOMAINS:
        common_inventory[domain] = {}
        for split, durations in zip(SPLITS, ([1.0, 2.0, 3.0, 4.0], [1.0 + 1e-10, 5.0], [6.0, 7.0])):
            directory = root / "input" / f"{domain}_{split}"
            directory.mkdir(parents=True)
            values = np.array([[duration, 10, 20, 30, 40] for duration in durations])
            table = pa.table({
                **{name: values[:, index] for index, name in enumerate(COMMON_FEATURES)},
                "label": np.arange(len(values)) % 2,
            })
            pq.write_table(table, directory / "part.parquet")
            common_inventory[domain][split] = inventory(directory)
    audit_path = root / "audit.json"
    audit_path.write_text(json.dumps({
        "quality_gate_passed": True,
        "common_inventory": common_inventory,
        "source_manifest": {
            "path": str(source_manifest),
            "sha256": data_revision.file_sha256(source_manifest),
        },
    }))
    return audit_path


class FrozenRevisionTests(unittest.TestCase):
    def test_collision_refit_preserves_flows_and_uses_final_source_train(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            audit_path = write_audited_inputs(root)
            with patch.object(freezing, "ROOT", root), redirect_stdout(io.StringIO()):
                manifest_path = freezing.freeze(
                    audit_path, max_relocated_fraction=1, max_class_prevalence_change=1,
                )
            manifest = json.loads(manifest_path.read_text())
            snapshot = manifest_path.parent
            relocation = json.loads((snapshot / "relocation_audit.json").read_text())
            for domain in DOMAINS:
                self.assertEqual(relocation["domains"][domain]["relocated_unique_rows"], 1)
                total_rows = sum(
                    pq.read_table(snapshot / "common" / f"{domain}_{split}").num_rows
                    for split in SPLITS
                )
                self.assertEqual(total_rows, 8)
                common_labels = np.concatenate([
                    pq.read_table(snapshot / "common" / f"{domain}_{split}")["label"].to_numpy()
                    for split in SPLITS
                ])
                np.testing.assert_array_equal(np.bincount(common_labels), [4, 4])

            for direction, source in (("unsw_to_cicids", "unsw"), ("cicids_to_unsw", "cicids")):
                train = pq.read_table(snapshot / "common" / f"{source}_train")
                source_train = np.column_stack([train[name].to_numpy() for name in COMMON_FEATURES])
                expected_processor = proposal_processor().fit(source_train)
                saved_processor = joblib.load(root / "models/canonical" / direction / "preprocessor.joblib")
                np.testing.assert_array_equal(
                    saved_processor.named_steps["imputer"].statistics_,
                    expected_processor.named_steps["imputer"].statistics_,
                )
                np.testing.assert_array_equal(
                    saved_processor.named_steps["scaler"].center_,
                    expected_processor.named_steps["scaler"].center_,
                )
                for domain in DOMAINS:
                    split_vectors = []
                    for split in SPLITS:
                        table = pq.read_table(snapshot / "prepared" / direction / f"{domain}_{split}")
                        matrix = vector_matrix(table["features"].combine_chunks())
                        split_vectors.append({tuple(row) for row in matrix})
                    for index in range(len(SPLITS)):
                        for other in range(index + 1, len(SPLITS)):
                            self.assertFalse(split_vectors[index] & split_vectors[other])

            with patch.dict(os.environ, {"KLTN_DATA_MANIFEST": str(manifest_path)}):
                self.assertEqual(data_revision.verify_revision()["data_revision"], "canonical")
                extra_file = snapshot / "common/unsw_train/extra.parquet"
                extra_file.write_bytes(b"unexpected")
                with self.assertRaisesRegex(ValueError, "inventory changed"):
                    data_revision.verify_revision()
            with patch.object(freezing, "ROOT", root):
                with self.assertRaises(FileExistsError):
                    freezing.freeze(audit_path)

    def test_relocation_limit_rejects_snapshot_before_manifest_is_frozen(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            audit_path = write_audited_inputs(root)
            with patch.object(freezing, "ROOT", root), redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, "design limits"):
                    freezing.freeze(audit_path, max_relocated_fraction=0)
            snapshot = root / "data/revisions/canonical"
            self.assertFalse((snapshot / "manifest.json").exists())
            report = json.loads((snapshot / "relocation_audit.json").read_text())
            self.assertEqual(report["status"], "design_limits_exceeded_before_final_refit")


class ParquetTrainingStreamTests(unittest.TestCase):
    def test_arrow_slice_does_not_include_values_outside_the_slice(self):
        rows = np.arange(20, dtype=np.float32).reshape(4, 5)
        for list_type in (pa.list_(pa.float32(), 5), pa.list_(pa.float32()), pa.large_list(pa.float32())):
            vectors = pa.array(rows.tolist(), type=list_type).slice(1, 2)
            np.testing.assert_array_equal(vector_matrix(vectors), rows[1:3])
        with self.assertRaisesRegex(ValueError, "Expected 5 features"):
            vector_matrix(pa.array([[1.0, 2.0]]))

    def test_batch_remainders_cross_files_and_shuffle_is_reproducible(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            expected = np.arange(35, dtype=np.float32).reshape(7, 5)
            for name, values in (("a", expected[:3]), ("b", expected[3:])):
                pq.write_table(pa.table({
                    "features": pa.array(values.tolist(), type=pa.list_(pa.float32(), 5)),
                    "label": values[:, 0].astype(np.int64) % 2,
                }), root / f"{name}.parquet")
            batches = list(iter_parquet_batches(root, 4, False, 42, True))
            self.assertEqual([len(features) for features, _ in batches], [4, 3])
            np.testing.assert_array_equal(np.concatenate([features.numpy() for features, _ in batches]), expected)
            np.testing.assert_array_equal(np.concatenate([labels.numpy() for _, labels in batches]), expected[:, 0].astype(np.int64) % 2)
            dropped = list(iter_parquet_batches(root, 4, False, 42, False, drop_last=True))
            self.assertEqual([len(batch) for batch in dropped], [4])
            first = ParquetBatchStream(root, 4, True, 42, False)
            second = ParquetBatchStream(root, 4, True, 42, False)
            for _ in range(2):
                actual = np.concatenate([batch.numpy() for batch in first])
                repeated = np.concatenate([batch.numpy() for batch in second])
                np.testing.assert_array_equal(actual, repeated)
                np.testing.assert_array_equal(np.sort(actual[:, 0]), expected[:, 0])


if __name__ == "__main__":
    unittest.main()
