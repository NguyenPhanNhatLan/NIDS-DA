"""Independent classifier with frozen source representations and a target adapter."""
import copy
from contextlib import contextmanager

import torch
from torch import nn


class HDAV5DModel(nn.Module):
    def __init__(self, source_model, target_adapter):
        super().__init__()
        # Private copies prevent parameters AND training-mode changes leaking to teachers.
        self.source_encoder = copy.deepcopy(source_model)
        self.source_encoder.requires_grad_(False)
        self.source_encoder.eval()
        self.adapter = copy.deepcopy(target_adapter)
        self.adapter.requires_grad_(True)
        self.classifier = copy.deepcopy(source_model.classifier)
        self.classifier.requires_grad_(True)

    def train(self, mode=True):
        super().train(mode)
        self.source_encoder.eval()  # Includes source dropout and both frozen BNs.
        return self

    def source_representations(self, x):
        with torch.no_grad():
            hidden = self.source_encoder.encode_hidden(x)
            latent = self.shared_latent(hidden)
        return hidden, latent

    def shared_latent(self, hidden):
        # Frozen weights, but preserve the gradient path to the target adapter.
        return torch.relu(self.source_encoder.bn2(self.source_encoder.fc2(hidden)))

    def forward_source(self, x):
        _, latent = self.source_representations(x)
        return latent, self.classifier(latent)  # Classifier is outside no_grad.

    def forward_target(self, x):
        latent = self.shared_latent(self.adapter(x))
        return latent, self.classifier(latent)

    def forward(self, x):
        return self.forward_target(x)

    @contextmanager
    def balanced_adapter_batch(self):
        bns = [m for m in self.adapter.modules()
               if isinstance(m, nn.modules.batchnorm._BatchNorm)]
        modes = [m.training for m in bns]
        try:
            for bn in bns:
                bn.eval()
            yield
        finally:
            for bn, mode in zip(bns, modes):
                bn.train(mode)

    def optimizer_groups(self, adapter_lr, classifier_lr):
        return [{"params": self.adapter.parameters(), "lr": adapter_lr},
                {"params": self.classifier.parameters(), "lr": classifier_lr}]
