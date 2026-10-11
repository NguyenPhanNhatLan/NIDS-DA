"""Separate SupCon projection; BaselineMLP checkpoint keys stay unchanged."""
from torch import nn
from torch.nn import functional as F


class ContrastiveProjection(nn.Module):
    def __init__(self, input_dim=168, output_dim=64):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, latent):
        return F.normalize(self.linear(latent), p=2, dim=1)
