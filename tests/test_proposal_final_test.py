import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from sklearn.metrics import average_precision_score

from evaluation import proposal_final_test as final_test
from evaluation.baseline import collect_scores, select_f1_threshold
from models.baseline import BaselineMLP
from training.proposal_data import ParquetBatchStream, split_sha256


def write_split(path, x, y):
    path.mkdir(parents=True)
    vectors = pa.FixedSizeListArray.from_arrays(pa.array(x.ravel()), 5)
    pq.write_table(pa.table({"features": vectors, "label": pa.array(y)}), path / "data.parquet")


class ProposalFinalTestTests(unittest.TestCase):
    def test_frozen_source_and_adapted_checkpoints_use_target_test_only_at_end(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            direction = "unsw_to_cicids"
            base = root / "data/features/proposal_v2" / direction
            rng = np.random.default_rng(7)
            x = rng.normal(size=(12, 5)).astype(np.float32)
            y = np.array([0, 1] * 6, dtype=np.int64)
            for name in ("unsw_train", "unsw_val", "cicids_train", "cicids_val", "cicids_test"):
                write_split(base / name, x, y)
            processor = root / "models/proposal_v2" / direction / "preprocessor.joblib"
            processor.parent.mkdir(parents=True)
            processor.write_bytes(b"frozen processor")
            schema = root / "configs/common_features_v2.json"
            schema.parent.mkdir(parents=True)
            schema.write_bytes(b"fixed common schema")
            prepared = {"source_train": split_sha256(base / "unsw_train"),
                        "source_val": split_sha256(base / "unsw_val"),
                        "target_train": split_sha256(base / "cicids_train"),
                        "target_val": split_sha256(base / "cicids_val")}
            model = BaselineMLP(5)
            labels, scores = collect_scores(model, ParquetBatchStream(base / "unsw_val", 1024, False, 42, True))
            ap = average_precision_score(labels, scores)
            threshold = select_f1_threshold(labels, scores)
            source_cp = root / "models/proposal_v2/source_only_target_val" / direction / "seed42.pt"
            source_result = root / "results/proposal_v2/source_only_target_val" / direction / "seed42.json"
            source_cp.parent.mkdir(parents=True)
            source_result.parent.mkdir(parents=True)
            shared = {"direction": direction, "seed": 42,
                      "features": list(final_test.COMMON_FEATURES),
                      "common_feature_config_sha256": final_test.sha256(schema),
                      "preprocessor_sha256": final_test.sha256(processor),
                      "prepared_split_sha256": prepared,
                      "best_epoch": 1, "best_source_val_ap": ap}
            torch.save({**shared, "input_dim": 5, "model_state_dict": model.state_dict()}, source_cp)
            source_data = {**shared, "feature_count": 5, "checkpoint": str(source_cp),
                           "target_development_split": "cicids_val",
                           "threshold_from_source_val": threshold}
            source_result.write_text(json.dumps(source_data))

            adapted_cp = root / "adapted.pt"
            adapted_result = root / "adapted.json"
            config = root / "configs/proposal_mmd_v2.json"
            config.write_bytes(b"frozen MMD config")
            adapted_shared = {**shared, "method": "marginal_mmd",
                              "config_sha256": final_test.sha256(config),
                              "source_checkpoint": str(source_cp),
                              "source_checkpoint_sha256": final_test.sha256(source_cp)}
            torch.save({**adapted_shared, "input_dim": 5, "model_state_dict": model.state_dict()}, adapted_cp)
            adapted_result.write_text(json.dumps({**adapted_shared, "feature_count": 5,
                "checkpoint": str(adapted_cp), "target_development_split": "cicids_val",
                "threshold": threshold}))

            with patch.object(final_test, "ROOT", root), \
                 patch.object(final_test, "OUTPUT_ROOT", root / "final"), \
                 patch.dict(final_test.CONFIGS, {"marginal_mmd": config}), \
                 patch.object(final_test, "output_paths", return_value=(adapted_cp, adapted_result)):
                for method in ("source_only", "marginal_mmd"):
                    result = final_test.run(direction, method, 42)
                    self.assertEqual(result["target_test_split"], "cicids_test")
                    self.assertIn("macro_f1", result["target_test"])
                    self.assertEqual(result["threshold_from_source_val"], threshold)

                source_data["threshold_from_source_val"] = threshold + 0.25
                source_result.write_text(json.dumps(source_data))
                with patch.object(final_test, "OUTPUT_ROOT", root / "final_stale"), \
                     patch.object(final_test, "collect_scores", wraps=final_test.collect_scores) as scores_spy:
                    with self.assertRaisesRegex(ValueError, "source-val AP/threshold"):
                        final_test.run(direction, "source_only", 42)
                    self.assertEqual(scores_spy.call_count, 1)

    def test_rejects_old_ten_feature_checkpoint(self):
        artifact = {"direction": "unsw_to_cicids", "seed": 42,
                    "features": list(final_test.COMMON_FEATURES), "input_dim": 10}
        with self.assertRaisesRegex(ValueError, "feature schema mismatch"):
            final_test.check_identity(artifact, "unsw_to_cicids", 42)


if __name__ == "__main__":
    unittest.main()
