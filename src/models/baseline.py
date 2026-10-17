"""Shared MLP for source-only and domain-adaptation experiments."""

from torch import nn
from torch.nn import functional as F


class BaselineMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 168)
        self.bn2 = nn.BatchNorm1d(168)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Sequential(
            nn.Linear(168, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def encode_hidden(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        return self.dropout(x)

    def forward(self, x):
        hidden = self.encode_hidden(x)
        features = F.relu(self.bn2(self.fc2(hidden)))
        return features, self.classifier(features)
