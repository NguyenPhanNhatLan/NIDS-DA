import torch.nn as nn


class TargetEncoder(nn.Module):
    def __init__(self, input_dim=46):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 168),
            nn.BatchNorm1d(168),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layers(x)


class TargetModel(nn.Module):
    def __init__(self, target_encoder, source_classifier):
        super().__init__()

        self.encoder = target_encoder
        self.classifier = source_classifier

        for parameter in self.classifier.parameters():
            parameter.requires_grad = False

    def forward(self, x):
        features = self.encoder(x)
        logits = self.classifier(features)
        return features, logits