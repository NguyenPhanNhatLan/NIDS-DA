import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from training.proposal_class_aware import audit_pseudo_labels


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))

    def forward(self, features):
        score = features[:, 0] + self.weight
        return features, torch.stack((-score, score), dim=1)


class PseudoAuditTests(unittest.TestCase):
    def test_audit_reads_feature_only_target_train(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            features = np.zeros((4, 5), dtype=np.float32)
            features[:, 0] = [-2.0, -1.0, 0.0, 2.0]
            vectors = pa.FixedSizeListArray.from_arrays(pa.array(features.ravel()), 5)
            pq.write_table(pa.table({"features": vectors}), path / "data.parquet")
            result = audit_pseudo_labels(ToyModel(), path, confidence=0.8)
        self.assertEqual(result["target_rows"], 4)
        self.assertEqual(result["accepted"], 3)
        self.assertEqual(result["pseudo_normal"], 2)
        self.assertEqual(result["pseudo_attack"], 1)
        self.assertAlmostEqual(result["accepted_rate"], 0.75)


if __name__ == "__main__":
    unittest.main()
