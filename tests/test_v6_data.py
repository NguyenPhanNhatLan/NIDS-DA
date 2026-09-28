from pathlib import Path
import tempfile
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from training import v6_data
from training.adaptation import make_unlabeled_loader
from training.baseline import make_loader, set_seed
from training.hda_v4 import make_teacher_loader


class V6DataTests(unittest.TestCase):
    def test_identical_batches_and_random_state(self):
        with tempfile.TemporaryDirectory() as directory:
            for index, rows in enumerate((8301, 503)):
                features = np.arange(rows * 3, dtype=np.float32).reshape(rows, 3)
                labels = (np.arange(rows) % 2).tolist()
                pq.write_table(pa.table({"features": features.tolist(), "label": labels}),
                               Path(directory) / f"part{index}.parquet")
            for labeled, training in ((True, True), (True, False), (False, True), (False, False)):
                with self.subTest(labeled=labeled, training=training):
                    if labeled:
                        old = make_loader(directory, 3, 257, training)
                        new = v6_data.make_loader(directory, 3, 257, training)
                    elif training:
                        old = make_unlabeled_loader(directory, 3, 257)
                        new = v6_data.make_unlabeled_loader(directory, 3, 257)
                    else:
                        old = make_teacher_loader(directory, 3, 257)
                        new = v6_data.make_teacher_loader(directory, 3, 257)
                    for epoch in range(2):
                        set_seed(42 + epoch)
                        expected = list(old)
                        expected_rng = torch.get_rng_state()
                        set_seed(42 + epoch)
                        actual = list(new)
                        self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
                        self.assertEqual(len(actual), len(expected))
                        for a, b in zip(actual, expected):
                            if labeled:
                                self.assertTrue(torch.equal(a[0], b[0]))
                                self.assertTrue(torch.equal(a[1], b[1]))
                            else:
                                self.assertTrue(torch.equal(a, b))

    def test_target_without_labels_and_invalid_dimensions(self):
        with tempfile.TemporaryDirectory() as directory:
            pq.write_table(pa.table({"features": [[1., 2.], [3., 4.], [5., 6.]]}),
                           Path(directory) / "rows.parquet")
            self.assertEqual(len(list(v6_data.make_unlabeled_loader(directory, 2, 2))), 1)
            self.assertEqual(len(list(v6_data.make_teacher_loader(directory, 2, 2))), 2)
            with self.assertRaises(ValueError):
                list(v6_data.make_teacher_loader(directory, 3, 2))


if __name__ == "__main__":
    unittest.main()
