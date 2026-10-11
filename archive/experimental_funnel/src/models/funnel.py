"""L2-normalized five-dimensional bottleneck for NIDS alignment."""
from torch import nn
from torch.nn import functional as F


class NIDSFunnelMLP(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.BatchNorm1d(64), nn.LeakyReLU(0.2),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.LeakyReLU(0.2),
            nn.Linear(32, 16), nn.BatchNorm1d(16), nn.LeakyReLU(0.2),
            nn.Linear(16, 5),
        )
        self.classifier = nn.Sequential(
            nn.Linear(5, 8), nn.LeakyReLU(0.2), nn.Linear(8, num_classes),
        )

    def forward(self, x):
        features = F.normalize(self.encoder(x), p=2, dim=1)
        return features, self.classifier(features)
