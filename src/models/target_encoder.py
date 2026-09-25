import torch.nn as nn


import torch
import torch.nn as nn


class TargetEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        latent_dim: int = 168,
        dropout: float = 0.3,
    ):
        super().__init__()

        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")

        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")

        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive.")

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim,
            ),

            nn.BatchNorm1d(
                hidden_dim
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                hidden_dim,
                latent_dim,
            ),

            nn.BatchNorm1d(
                latent_dim
            ),

            nn.ReLU(),
        )

    def forward(self, x):
        if x.ndim != 2:
            raise ValueError(
                f"Expected [batch, features], "
                f"got shape {tuple(x.shape)}"
            )

        if x.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected {self.input_dim} features, "
                f"got {x.shape[1]}"
            )

        return self.encoder(x)

class TargetModel(nn.Module):

    def __init__(
        self,
        target_encoder,
        source_classifier,
    ):
        super().__init__()

        self.encoder = target_encoder
        self.classifier = source_classifier

        # Freeze classifier learned from UNSW
        for parameter in (
            self.classifier.parameters()
        ):
            parameter.requires_grad = False

    def forward(self, x):

        representation = self.encoder(x)
    
        logits = self.classifier(
            representation
        )

        return representation, logits