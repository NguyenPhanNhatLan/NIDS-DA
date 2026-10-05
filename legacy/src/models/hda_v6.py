from torch import nn


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim=256, hidden_dim=384, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return x + self.net(self.norm(x))


class DomainStem(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
        )

    def forward(self, x):
        return self.net(x)


class SharedEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.Sequential(
            ResidualMLPBlock(),
            ResidualMLPBlock(),
        )
        self.bottleneck = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.LayerNorm(128),
        )

    def forward(self, x):
        return self.bottleneck(self.blocks(x))


class HDAV6Model(nn.Module):
    def __init__(self, source_dim, target_dim, architecture="residual"):
        super().__init__()
        if architecture == "plain":
            self.source_stem = nn.Sequential(
                nn.Linear(source_dim, 256), nn.LayerNorm(256),
                nn.ReLU(), nn.Dropout(0.3),
            )
            self.target_stem = nn.Sequential(
                nn.Linear(target_dim, 256), nn.LayerNorm(256), nn.ReLU(),
            )
            self.encoder = nn.Sequential(
                nn.Linear(256, 168), nn.LayerNorm(168), nn.ReLU(),
            )
            self.classifier = nn.Sequential(
                nn.Linear(168, 32), nn.ReLU(), nn.Linear(32, 2),
            )
        elif architecture == "residual":
            self.source_stem = DomainStem(source_dim)
            self.target_stem = DomainStem(target_dim)
            self.encoder = SharedEncoder()
            self.classifier = nn.Sequential(
                nn.Linear(128, 32), nn.GELU(),
                nn.Dropout(0.1), nn.Linear(32, 2),
            )
        else:
            raise ValueError("architecture must be plain or residual")

    def encode_hidden(self, x, domain="source"):
        if domain == "source":
            return self.source_stem(x)
        if domain == "target":
            return self.target_stem(x)
        raise ValueError("domain must be source or target")

    def forward(self, x, domain="source"):
        hidden = self.encode_hidden(x, domain)
        latent = self.encoder(hidden)
        return latent, self.classifier(latent)
