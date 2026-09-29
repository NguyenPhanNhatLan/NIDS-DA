import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch
from torch.nn import functional as F

from models.baseline import BaselineMLP
from models.hda_v1 import HDAV1Model
from models.hda_v5d import HDAV5DModel
from training.hda_v5d import baseline_class_weights, checkpoint_path, load_context, train_model, preflight, run
from evaluation.hda_v5d import operating_metrics, fit, validate_v5b_code_hashes
from evaluation.hda_v5b_calibration import payload_hash


class V5dTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.source = BaselineMLP(4).eval()
        self.teacher = HDAV1Model(3, self.source).eval().requires_grad_(False)
        self.model = HDAV5DModel(self.source, self.teacher.adapter)

    def test_only_reviewed_calibration_code_transition_is_accepted(self):
        name = "src/evaluation/hda_v5b_calibration.py"
        old = "276158843ee2b875dd984f75841f75e6b72f1ed169f689f0f3e4be7db8f38651"
        new = "af7de97279b5e48fbcbbfc0961cf807e4d66ae6ce9728e2f2fadfafda266b6db"
        frozen = {name: old, "src/evaluation/calibration.py": "unchanged"}
        current = {**frozen, name: new}
        self.assertEqual(validate_v5b_code_hashes(frozen, frozen), [])
        self.assertEqual(validate_v5b_code_hashes(frozen, current), [name])
        self.assertEqual(frozen[name], old)  # No rewriting producer provenance.
        for invalid in ({**current, name: "unknown-revision"},
                        {**current, "src/evaluation/calibration.py": "changed-math"},
                        {name: new}, {**current, "unexpected.py": "new"}):
            with self.assertRaisesRegex(ValueError, "code changed"):
                validate_v5b_code_hashes(frozen, invalid)
        with self.assertRaises(ValueError):
            validate_v5b_code_hashes(current, frozen)  # No blanket bidirectional allowance.

    def test_preflight_verifies_artifact_and_returns_same_context(self):
        config = {"teacher_seed": 42, "training_seed": 43, "classifier_only": True,
                  "calibration_max_fpr": .02}
        context = (config, {}, {}, self.source, self.teacher, self.teacher)
        with (patch("training.hda_v5d.load_context", return_value=context) as load,
              patch("evaluation.hda_v5d.load_v5b_calibration", return_value={}) as verify):
            self.assertIs(preflight("config.json", 43), context)
        load.assert_called_once_with("config.json", 43)
        verify.assert_called_once_with(config, {}, {})

    def test_bad_preflight_stops_before_pools_or_training(self):
        context = ({}, {}, {}, self.source, self.teacher, self.teacher)
        with (patch("training.hda_v5d.load_context", return_value=context),
              patch("evaluation.hda_v5d.load_v5b_calibration", side_effect=ValueError("artifact mismatch")),
              patch("training.hda_v5d.build_frozen_target_pools") as pools,
              patch("training.hda_v5d.build_source_pools") as source_pools,
              patch("training.hda_v5d.train_model") as train):
            with self.assertRaisesRegex(ValueError, "artifact mismatch"):
                run("config.json")
            pools.assert_not_called()
            source_pools.assert_not_called()
            train.assert_not_called()

    def test_classifier_is_a_copy_not_shared_storage(self):
        before = self.source.classifier[0].weight.detach().clone()
        self.assertNotEqual(self.model.classifier[0].weight.data_ptr(),
                            self.source.classifier[0].weight.data_ptr())
        with torch.no_grad():
            self.model.classifier[0].weight[0, 0].add_(1.)
        self.assertTrue(torch.equal(before, self.source.classifier[0].weight))
        self.assertTrue(torch.equal(before, self.teacher.classifier[0].weight))
        for actual, expected in zip(self.model.adapter.parameters(), self.teacher.adapter.parameters()):
            self.assertTrue(torch.equal(actual, expected))
            self.assertNotEqual(actual.data_ptr(), expected.data_ptr())

    def test_source_ce_trains_only_private_classifier(self):
        self.model.train()
        frozen_before = {k: v.clone() for k, v in self.model.source_encoder.state_dict().items()}
        classifier_before = self.model.classifier[0].weight.detach().clone()
        _, logits = self.model.forward_source(torch.randn(8, 4))
        F.cross_entropy(logits, torch.tensor([0, 1] * 4), weight=baseline_class_weights([3, 7])).backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in self.model.classifier.parameters()))
        self.assertTrue(all(p.grad is None for p in self.model.source_encoder.parameters()))
        self.assertTrue(all(p.grad is None for p in self.model.adapter.parameters()))
        optimizer = torch.optim.Adam(self.model.optimizer_groups(1e-4, 1e-5))
        optimizer.step()
        self.assertFalse(torch.equal(classifier_before, self.model.classifier[0].weight))
        for k, v in self.model.source_encoder.state_dict().items():
            self.assertTrue(torch.equal(v, frozen_before[k]), k)

    def test_target_gradients_cross_frozen_shared_layers(self):
        self.model.train()
        _, logits = self.model.forward_target(torch.randn(8, 3))
        F.cross_entropy(logits, torch.tensor([0, 1] * 4)).backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in self.model.adapter.parameters()))
        self.assertTrue(all(p.grad is None for p in self.model.source_encoder.parameters()))

    def test_bn_policy_and_context_restore_on_exception(self):
        self.model.train()
        self.assertFalse(self.model.source_encoder.bn1.training)
        self.assertFalse(self.model.source_encoder.bn2.training)
        self.assertFalse(self.model.source_encoder.dropout.training)
        bn = self.model.adapter.layers[1]
        self.model.forward_target(torch.randn(8, 3))
        count = bn.num_batches_tracked.clone()
        with self.assertRaises(RuntimeError):
            with self.model.balanced_adapter_batch():
                self.assertFalse(bn.training)
                self.model.forward_target(torch.randn(8, 3))
                raise RuntimeError("test restore")
        self.assertTrue(bn.training)
        self.assertTrue(torch.equal(count, bn.num_batches_tracked))

    def test_optimizer_contains_exactly_adapter_and_classifier(self):
        optimizer = torch.optim.Adam(self.model.optimizer_groups(1e-4, 1e-5))
        actual = [id(p) for g in optimizer.param_groups for p in g["params"]]
        expected = {id(p) for p in self.model.adapter.parameters()} | {id(p) for p in self.model.classifier.parameters()}
        self.assertEqual(set(actual), expected)
        self.assertEqual(len(actual), len(expected))
        self.assertEqual([g["lr"] for g in optimizer.param_groups], [1e-4, 1e-5])

    def test_baseline_weighting(self):
        self.assertTrue(torch.allclose(baseline_class_weights([2, 8]), torch.tensor([2.5, 0.625])))
        for counts in ([0, 4], [-1, 4], [1], [1, float("nan")]):
            with self.assertRaises(ValueError):
                baseline_class_weights(counts)

    def test_classifier_only_freezes_adapter_weights_and_all_bn_buffers(self):
        model = HDAV5DModel(self.source, self.teacher.adapter, classifier_only=True)
        before = {k: v.clone() for k, v in model.adapter.state_dict().items()}
        classifier_before = model.classifier[0].weight.detach().clone()
        groups = model.optimizer_groups(1e-4, 1e-5)
        optimizer = torch.optim.Adam(groups)
        self.assertEqual({id(p) for g in optimizer.param_groups for p in g["params"]},
                         {id(p) for p in model.classifier.parameters()})
        for _ in range(2):
            model.train()
            self.assertFalse(model.adapter.training)
            self.assertFalse(model.adapter.layers[1].training)
            _, logits = model.forward_target(torch.randn(8, 3))
            with model.balanced_adapter_batch():
                model.forward_target(torch.randn(8, 3))
            optimizer.zero_grad()
            F.cross_entropy(logits, torch.tensor([0, 1] * 4)).backward()
            optimizer.step()
        for k, v in model.adapter.state_dict().items():
            self.assertTrue(torch.equal(v, before[k]), k)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.adapter.parameters()))
        self.assertFalse(torch.equal(classifier_before, model.classifier[0].weight))

    def test_affine_preserves_ranking_but_changes_operating_point(self):
        labels = torch.tensor([0, 0, 1, 1])
        margins = torch.tensor([-4., -3., -2., -1.], dtype=torch.float64)
        result = operating_metrics(labels, margins, {"a": 2., "b": 5.}, 0.)
        for name in ("pr_auc", "roc_auc"):
            self.assertEqual(result["raw"][name], result["calibrated"][name])
        self.assertGreater(result["calibrated"]["f1"], result["raw"]["f1"])
        for a in (0., -1., float("nan")):
            with self.assertRaises(ValueError):
                operating_metrics(labels, margins, {"a": a, "b": 0.}, 0.)

    def test_training_seed_does_not_select_another_teacher(self):
        reference = {"seed": 42, "checkpoint_sha256": "fixed-v5b",
                     "loss_weights": {"hidden": 1., "normal": .05, "attack": .02, "rank": .1}}
        protocol = {"target_data": {"adaptation_train": "data/features/cicids_train",
                                    "development": "data/features/cicids_test"}}
        context = (reference, protocol, {}, self.source, self.teacher, self.teacher)
        with patch("training.hda_v5d.load_frozen_models", return_value=context) as loader:
            paths = []
            for seed in (42, 43, 44):
                config, _, provenance, *_ = load_context("configs/hda_v5d.json", seed)
                self.assertEqual(provenance["teacher_seed"], 42)
                self.assertEqual(provenance["training_seed"], seed)
                paths.append(checkpoint_path(config))
            self.assertEqual(len(set(paths)), 3)
            self.assertTrue(all(call == loader.call_args_list[0] for call in loader.call_args_list))

    def test_fit_uses_frozen_v5b_and_unlabeled_v2_pools(self):
        with tempfile.TemporaryDirectory() as directory:
            config = {"training_seed": 43, "teacher_seed": 42, "classifier_only": True,
                      "calibration_dir": directory, "source_validation": "source-val",
                      "calibration_max_fpr": .02, "v5b_calibration": "frozen-v5b"}
            protocol = {"training": {"batch_size": 4},
                        "target_data": {"adaptation_train": "target-train", "development": "target-dev"}}
            reference = {"sha256": "pinned", "frozen": {"a": 1.2, "b": .3,
                         "source_margin_threshold": .7, "source_validation_metrics": {}}}
            source_margins = torch.tensor([-3., -2., 2., 3.], dtype=torch.float64)
            labels = torch.tensor([0, 0, 1, 1])
            pools = (torch.zeros(4, 3), torch.ones(4, 3), {"labels_used": False})
            with (patch("evaluation.hda_v5d.load_context", return_value=(config, protocol, {"source_dim": 4, "target_dim": 3}, self.source, self.teacher, self.teacher)),
                  patch("evaluation.hda_v5d.load_student", return_value=(self.model, {})),
                  patch("evaluation.hda_v5d.verify_training_data"),
                  patch("evaluation.hda_v5d.load_v5b_calibration", return_value=reference),
                  patch("evaluation.hda_v5d.dependencies", return_value={}),
                  patch("evaluation.hda_v5d.data_snapshot", return_value={}),
                  patch("evaluation.hda_v5d.make_loader", return_value=[]),
                  patch("evaluation.hda_v5d.collect_source_margins", return_value=(source_margins, labels)),
                  patch("evaluation.hda_v5d.build_frozen_target_pools", return_value=pools) as build,
                  patch("evaluation.hda_v5d.collect_margins", side_effect=[(torch.tensor([-5., -4.]), None), (torch.tensor([1., 2.]), None)]) as collect):
                fit("unused", 43)
            build.assert_called_once_with(self.teacher, protocol, 3, 4)
            self.assertEqual(collect.call_count, 2)
            self.assertTrue(all(call.args[0] is self.model for call in collect.call_args_list))
            artifact = json.loads((Path(directory) / "seed43.json").read_text())
            self.assertEqual(payload_hash(artifact["frozen"]), artifact["sha256"])
            frozen = artifact["frozen"]
            self.assertEqual(frozen["parameters"]["v5b"], {"a": 1.2, "b": .3})
            self.assertEqual(frozen["thresholds"]["v5b"], .7)
            self.assertGreater(frozen["parameters"]["v5d"]["a"], 0)
            self.assertFalse(frozen["target_labels_used_for_fit"])

    def test_source_ce_only_skips_alignment_teacher_and_adapter(self):
        model = HDAV5DModel(self.source, self.teacher.adapter, classifier_only=True)
        before = {k: v.clone() for k, v in model.adapter.state_dict().items()}
        classifier_before = model.classifier[0].weight.detach().clone()
        config = {"adapter_lr": 1e-4, "classifier_lr": 1e-5, "weight_decay": 1e-4,
                  "loss_weights": {"hidden": 0., "normal": 0., "attack": 0., "rank": 0., "source": 1.}}
        with (patch("training.hda_v5d.mmd_loss", side_effect=AssertionError("MMD must be skipped")),
              patch("training.hda_v5d.ranking_loss", side_effect=AssertionError("Ranking must be skipped")),
              patch.object(self.teacher, "forward", side_effect=AssertionError("Teacher must be skipped")),
              patch.object(model.adapter, "forward", side_effect=AssertionError("Adapter must be skipped"))):
            history = train_model(model, self.teacher,
                                  [(torch.randn(8, 4), torch.tensor([0, 1] * 4))],
                                  [torch.randn(8, 3)], None, None, baseline_class_weights([4, 4]),
                                  config, {"epochs": 1, "class_batch_size": 4})
        self.assertEqual(history[0]["loss"], history[0]["source"])
        for key in ("hidden", "normal", "attack", "rank"):
            self.assertEqual(history[0][key], 0.)
        for key, value in model.adapter.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]), key)
        self.assertFalse(torch.equal(classifier_before, model.classifier[0].weight))

    def test_training_step_preserves_teacher_source_and_frozen_buffers(self):
        teacher_before = {k: v.clone() for k, v in self.teacher.state_dict().items()}
        source_before = {k: v.clone() for k, v in self.source.state_dict().items()}
        frozen_before = {k: v.clone() for k, v in self.model.source_encoder.state_dict().items()}
        adapter_before = self.model.adapter.layers[0].weight.detach().clone()
        classifier_before = self.model.classifier[0].weight.detach().clone()
        x, y, target = torch.randn(8, 4), torch.tensor([0, 1] * 4), torch.randn(8, 3)
        _, latent = self.model.source_representations(x)
        config = {"adapter_lr": 1e-4, "classifier_lr": 1e-5, "weight_decay": 1e-4,
                  "loss_weights": {"hidden": 1., "normal": .05, "attack": .02, "rank": .1, "source": .1}}
        history = train_model(self.model, self.teacher, [(x, y)], [target],
                              {0: latent[y == 0], 1: latent[y == 1]},
                              {0: target[:4], 1: target[4:]}, baseline_class_weights([4, 4]),
                              config, {"class_batch_size": 4, "epochs": 1})
        for model, before in ((self.teacher, teacher_before), (self.source, source_before),
                              (self.model.source_encoder, frozen_before)):
            for k, v in model.state_dict().items():
                self.assertTrue(torch.equal(v, before[k]), k)
        self.assertFalse(torch.equal(adapter_before, self.model.adapter.layers[0].weight))
        self.assertFalse(torch.equal(classifier_before, self.model.classifier[0].weight))
        self.assertEqual(self.model.adapter.layers[1].num_batches_tracked.item(), 1)
        self.assertTrue(all(p.grad is None for p in self.teacher.parameters()))
        row = history[0]
        expected_loss = sum(config["loss_weights"][k] * row[k] for k in config["loss_weights"])
        self.assertAlmostEqual(row["loss"], expected_loss, places=5)


if __name__ == "__main__":
    unittest.main()
