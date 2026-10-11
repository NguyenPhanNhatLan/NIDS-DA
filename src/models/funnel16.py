"""Separate classification and alignment representations."""
from torch import nn
from torch.nn import functional as F


class NIDSFunnel16(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.BatchNorm1d(64), nn.LeakyReLU(0.2),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.LeakyReLU(0.2),
            nn.Linear(32, 16),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16, 8), nn.LeakyReLU(0.2), nn.Linear(8, num_classes),
        )

    def forward(self, x):
        z = self.encoder(x)
        return z, F.normalize(z, dim=1), self.classifier(z)
