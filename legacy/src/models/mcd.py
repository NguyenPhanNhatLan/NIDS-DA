"""Two-head MCD model compatible with the Common-5 proposal pipeline."""

from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn
import torch.nn.functional as F


class MCDEncoder(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 168)
        self.bn2 = nn.BatchNorm1d(168)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.bn2(x)
        return F.relu(x)


class MCDClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(168, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, z):
        return self.net(z)


class MCDModel(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.encoder = MCDEncoder(input_dim)
        self.classifier1 = MCDClassifier()
        self.classifier2 = MCDClassifier()

    def encode(self, x):
        return self.encoder(x)

    def forward_heads(self, z):
        return self.classifier1(z), self.classifier2(z)

    def forward(self, x):
        z = self.encoder(x)
        logits1, logits2 = self.forward_heads(z)
        return z, 0.5 * (logits1 + logits2)

    @torch.no_grad()
    def predict_proba(self, x):
        z = self.encoder(x)
        logits1, logits2 = self.forward_heads(z)
        p1 = torch.softmax(logits1, dim=1)
        p2 = torch.softmax(logits2, dim=1)
        return 0.5 * (p1 + p2)


def load_from_baseline_checkpoint(
    model: MCDModel,
    baseline_state_dict: dict,
    classifier2_noise_std: float = 1e-3,
):
    encoder_state = {
        "fc1.weight": baseline_state_dict["fc1.weight"],
        "fc1.bias": baseline_state_dict["fc1.bias"],
        "bn1.weight": baseline_state_dict["bn1.weight"],
        "bn1.bias": baseline_state_dict["bn1.bias"],
        "bn1.running_mean": baseline_state_dict["bn1.running_mean"],
        "bn1.running_var": baseline_state_dict["bn1.running_var"],
        "bn1.num_batches_tracked": baseline_state_dict["bn1.num_batches_tracked"],
        "fc2.weight": baseline_state_dict["fc2.weight"],
        "fc2.bias": baseline_state_dict["fc2.bias"],
        "bn2.weight": baseline_state_dict["bn2.weight"],
        "bn2.bias": baseline_state_dict["bn2.bias"],
        "bn2.running_mean": baseline_state_dict["bn2.running_mean"],
        "bn2.running_var": baseline_state_dict["bn2.running_var"],
        "bn2.num_batches_tracked": baseline_state_dict["bn2.num_batches_tracked"],
    }
    model.encoder.load_state_dict(encoder_state)

    c1_state = {
        "net.0.weight": baseline_state_dict["classifier.0.weight"],
        "net.0.bias": baseline_state_dict["classifier.0.bias"],
        "net.2.weight": baseline_state_dict["classifier.2.weight"],
        "net.2.bias": baseline_state_dict["classifier.2.bias"],
    }
    model.classifier1.load_state_dict(c1_state)
    model.classifier2.load_state_dict(deepcopy(c1_state))

    if classifier2_noise_std < 0:
        raise ValueError("classifier2_noise_std must be >= 0")

    if classifier2_noise_std > 0:
        with torch.no_grad():
            for parameter in model.classifier2.parameters():
                parameter.add_(classifier2_noise_std * torch.randn_like(parameter))

    return model
