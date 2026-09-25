"""HDAv1: CICIDS adapter -> frozen UNSW tail -> frozen classifier."""

import torch
from torch import nn


class TargetAdapter(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

    def forward(self, x):
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError(f"Adapter expects [batch, {self.input_dim}], got {tuple(x.shape)}")
        return self.layers(x)


class SharedTargetEncoder(nn.Module):
    def __init__(self, adapter, source_model):
        super().__init__()
        self.adapter = adapter
        # These are the exact modules from the pretrained UNSW checkpoint.
        self.shared_fc2 = source_model.fc2
        self.shared_bn2 = source_model.bn2

    def forward(self, x):
        x = self.adapter(x)
        x = self.shared_fc2(x)
        x = self.shared_bn2(x)
        return torch.relu(x)


class HDAV1Model(nn.Module):
    def __init__(self, input_dim, source_model):
        super().__init__()
        self.encoder = SharedTargetEncoder(TargetAdapter(input_dim), source_model)
        self.classifier = source_model.classifier

        for module in (self.encoder.shared_fc2, self.encoder.shared_bn2, self.classifier):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad = False

    @property
    def adapter(self):
        return self.encoder.adapter

    def forward(self, x):
        latent = self.encoder(x)
        logits = self.classifier(latent)
        return latent, logits
