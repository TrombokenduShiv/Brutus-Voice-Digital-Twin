from __future__ import annotations

import torch
from torch import nn


class AllophoneModel(nn.Module):
    """Predicts target-speaker phoneme realization statistics from HDVR hidden states."""

    def __init__(self, hidden_dim: int = 512, feature_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden)
