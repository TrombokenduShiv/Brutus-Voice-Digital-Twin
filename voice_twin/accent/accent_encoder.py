from __future__ import annotations

import torch
from torch import nn


class AccentFeatureEncoder(nn.Module):
    def __init__(self, feature_dim: int = 8, output_dim: int = 192):
        super().__init__()
        self.norm = nn.LayerNorm(feature_dim)
        self.net = nn.Sequential(
            nn.Linear(feature_dim, output_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(self.norm(features))
