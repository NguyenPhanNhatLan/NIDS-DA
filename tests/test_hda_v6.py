import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch import nn

from evaluation.hda_v6 import collect_predictions
from models.hda_v6 import HDAV6Model, ResidualMLPBlock
from training.adaptation import make_unlabeled_loader
from training.baseline import set_seed
from training.hda_v6 import checkpoint_path, configure_training, train_alignment


ROOT = Path(__file__).resolve().parents[1]


class HDAV6Tests(unittest.TestCase):
    def test_architecture_and_batch_independence(self):
        for architecture, latent_dim in (("plain", 168), ("residual", 128)):
            model = HDAV6Model(4, 3, architecture).eval()
            self.assertFalse(any(isinstance(layer, nn.modules.batchnorm._BatchNorm)
                                 for layer in model.modules()))
            for domain, input_dim in (("source", 4), ("target", 3)):
                features = torch.randn(5, input_dim)
                latent, logits = model(features, domain)
                self.assertEqual(latent.shape, (5, latent_dim))
                self.assertEqual(logits.shape, (5, 2))
                torch.testing.assert_close(model(features[:1], domain)[1], logits[:1], atol=1e-6, rtol=1e-5)

    def test_residual_identity(self):
        block = ResidualMLPBlock().eval()
        for parameter in block.net.parameters():
            nn.init.zeros_(parameter)
        features = torch.randn(4, 256)
        torch.testing.assert_close(block(features), features)

    def test_freeze_modes_and_asymmetric_loss(self):
        config = json.loads((ROOT / "configs/hda_v6.json").read_text())
        for version, settings in config["variants"].items():
            with self.subTest(version=version):
                set_seed(42)
                model = HDAV6Model(4, 3, settings["architecture"])
                before = {key: value.clone() for key, value in model.state_dict().items()}
                source_batches = [(torch.randn(8, 4), torch.tensor([0, 1] * 4))]
                target_batches = [torch.randn(8, 3)]
                source_pools = {label: torch.randn(12, 4) for label in (0, 1)}
                target_pools = {label: torch.randn(12, 3) for label in (0, 1)}
                with contextlib.redirect_stdout(io.StringIO()):
                    history = train_alignment(
                        model, source_batches, target_batches, settings, 1, 0.001, 0.0001,
                        4, source_pools, target_pools, source_counts=[90, 10],
                    )
                row = history[0]
                expected = sum(settings[name] * row[name] for name in ("source_ce", "marginal", "normal", "attack"))
                self.assertAlmostEqual(row["loss"], expected, places=6)
                for module in ("source_stem", "target_stem", "encoder", "classifier"):
                    changed = any(not torch.equal(value, model.state_dict()[key])
                                  for key, value in before.items() if key.startswith(module + "."))
                    expected_change = module == "target_stem" or (settings["train_shared"] and module == "encoder")
                    self.assertEqual(changed, expected_change)
                self.assertEqual(row["source_ce"] > 0, version == "v6c")

    def test_weighted_ce_anchor_and_learning_rates(self):
        set_seed(42)
        model = HDAV6Model(4, 3)
        configure_training(model, True)
        self.assertFalse(model.source_stem.training)
        self.assertFalse(model.classifier.training)
        self.assertTrue(model.target_stem.training)
        self.assertTrue(model.encoder.training)
        before = {key: value.clone() for key, value in model.state_dict().items()}
        settings = {"train_shared": True, "source_ce": 1.0,
                    "marginal": 0.0, "normal": 0.0, "attack": 0.0}
        source_loader = [(torch.randn(8, 4), torch.tensor([0, 1] * 4))]
        target_loader = [torch.randn(8, 3)]
        with patch("training.hda_v6.nn.CrossEntropyLoss", wraps=nn.CrossEntropyLoss) as criterion, \
                patch("training.hda_v6.torch.optim.Adam", wraps=torch.optim.Adam) as optimizer, \
                contextlib.redirect_stdout(io.StringIO()):
            train_alignment(model, source_loader, target_loader, settings, 1, 0.001, 0.0001,
                            source_counts=[90, 10], shared_lr=0.0001)
        torch.testing.assert_close(criterion.call_args.kwargs["weight"], torch.tensor([100 / 180, 5.0]))
        groups = optimizer.call_args.args[0]
        self.assertEqual([group["lr"] for group in groups], [0.001, 0.0001])
        self.assertTrue(any(not torch.equal(value, model.state_dict()[key])
                            for key, value in before.items() if key.startswith("encoder.")))
        for name in ("source_stem", "classifier"):
            for key, value in before.items():
                if key.startswith(name + "."):
                    torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
            self.assertTrue(all(parameter.grad is None for parameter in getattr(model, name).parameters()))

    def test_source_ce_requires_valid_counts(self):
        settings = {"train_shared": True, "source_ce": 1.0,
                    "marginal": 0.1, "normal": 0.0, "attack": 0.0}
        for counts in (None, [10, 0], [10], [10, float("nan")]):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                train_alignment(HDAV6Model(4, 3), [], [], settings, 1, 0.001, 0.0001,
                                source_counts=counts)

    def test_target_training_without_label_column(self):
        with tempfile.TemporaryDirectory() as directory:
            features = torch.randn(8, 3)
            pq.write_table(pa.table({"features": features.tolist()}), Path(directory) / "rows.parquet")
            target_loader = make_unlabeled_loader(directory, 3, batch_size=4)
            model = HDAV6Model(4, 3, "plain")
            settings = {"train_shared": False, "source_ce": 0.0,
                        "marginal": 1.0, "normal": 0.0, "attack": 0.0}
            source_loader = [(torch.randn(4, 4), torch.tensor([0, 1, 0, 1]))]
            with contextlib.redirect_stdout(io.StringIO()):
                history = train_alignment(model, source_loader, target_loader, settings, 1, 0.001, 0.0001)
            self.assertEqual(len(history), 1)

    def test_prediction_collection_and_shared_warmup(self):
        model = HDAV6Model(4, 3)
        configure_training(model, True)
        labels, scores, logits = collect_predictions(
            model, [(torch.randn(4, 3), torch.tensor([0, 1, 0, 1]))], "target",
        )
        self.assertEqual(labels.tolist(), [0, 1, 0, 1])
        self.assertEqual(scores.shape, (4,))
        self.assertEqual(logits.shape, (4, 2))
        self.assertFalse(model.training)
        config = json.loads((ROOT / "configs/hda_v6.json").read_text())
        for stage in ("source", "warmup"):
            self.assertEqual(checkpoint_path(config, "v6b", stage, 42),
                             checkpoint_path(config, "v6c", stage, 42))


if __name__ == "__main__":
    unittest.main()
