from copy import deepcopy

import torch
from torch import nn


class ResidualTargetCorrection(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, dim),
        )
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, hidden):
        return hidden + self.alpha * self.block(hidden)


class ResidualHDAModel(nn.Module):
    def __init__(self, source, v2_adapter):
        super().__init__()
        self.source = source
        self.adapter = deepcopy(v2_adapter)
        for parameter in self.parameters():
            parameter.requires_grad = False
        self.correction = ResidualTargetCorrection()
        self.eval()

    def train(self, mode=True):
        super().train(mode)
        self.source.eval()
        self.adapter.eval()
        return self

    def encode_base(self, features):
        with torch.no_grad():
            return self.adapter(features)

    def classify_hidden(self, hidden):
        latent = torch.relu(self.source.bn2(self.source.fc2(hidden)))
        return latent, self.source.classifier(latent)

    def forward(self, features):
        hidden = self.correction(self.encode_base(features))
        return self.classify_hidden(hidden)
