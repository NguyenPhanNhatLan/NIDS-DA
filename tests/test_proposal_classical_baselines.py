import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from experiments import proposal_classical_baselines as classical


def write_split(path, features, labels):
    path.mkdir(parents=True)
    vectors = pa.FixedSizeListArray.from_arrays(pa.array(features.ravel()), 5)
    pq.write_table(pa.table({"features": vectors, "label": pa.array(labels)}), path / "data.parquet")


class ProposalClassicalBaselineTests(unittest.TestCase):
    def test_lr_uses_source_train_and_source_val_threshold(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base = root / "data/features/proposal_v2/unsw_to_cicids"
            x = np.zeros((40, 5), dtype=np.float32)
            x[:, 0] = np.tile([-2.0, 2.0], 20)
            y = np.tile([0, 1], 20)
            write_split(base / "unsw_train", x, y)
            write_split(base / "unsw_val", x[:20], y[:20])
            write_split(base / "cicids_val", x[20:], y[20:])
            processor = root / "models/proposal_v2/unsw_to_cicids/preprocessor.joblib"
            processor.parent.mkdir(parents=True)
            processor.write_bytes(b"source fitted processor")
            with patch.object(classical, "ROOT", root), \
                 patch.object(classical, "RESULT_ROOT", root / "results"):
                result = classical.run("unsw_to_cicids", "logistic_regression", 42, 20)
            self.assertEqual(result["train_rows_sampled"], 20)
            self.assertEqual(result["target_development_split"], "cicids_val")
            self.assertEqual(result["feature_count"], 5)
            self.assertIn("macro_f1", result["source_val"])
            self.assertIn("pr_auc", result["target_development"])
            self.assertTrue((root / "results/logistic_regression/unsw_to_cicids/seed42.json").is_file())


if __name__ == "__main__":
    unittest.main()
