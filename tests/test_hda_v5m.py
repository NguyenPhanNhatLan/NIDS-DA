"""V5m assembly invariants; no training or real data required."""
import unittest

import torch

from models.baseline import BaselineMLP
from models.hda_v1 import HDAV1Model
from training.hda_v5m import build_v5m


class V5mTests(unittest.TestCase):
    def test_loads_only_v5d_adapter_and_keeps_source_classifier(self):
        torch.manual_seed(42)
        source = BaselineMLP(4).eval()
        donor = HDAV1Model(3, source).eval()
        with torch.no_grad():
            donor.adapter.layers[0].weight.add_(.25)
        before = {k: v.clone() for k, v in source.state_dict().items()}
        # Deliberately invalid classifier state proves it is never loaded.
        checkpoint = {"target_adapter_state_dict": donor.adapter.state_dict(),
                      "classifier_state_dict": {"must_not_load": torch.tensor(float("nan"))}}
        model = build_v5m(source, checkpoint, 3)
        for k, value in model.adapter.state_dict().items():
            torch.testing.assert_close(value, donor.adapter.state_dict()[k], rtol=0, atol=0)
        for k, value in model.classifier.state_dict().items():
            torch.testing.assert_close(value, source.classifier.state_dict()[k], rtol=0, atol=0)
        with torch.no_grad():
            x = torch.randn(8, 3)
            torch.testing.assert_close(model(x)[1], donor(x)[1], rtol=0, atol=0)
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))
        for k, value in source.state_dict().items():
            torch.testing.assert_close(value, before[k], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
