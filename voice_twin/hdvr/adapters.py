from __future__ import annotations

import torch
from torch import nn


class SpeakerLoRAAdapter(nn.Module):
    """Low-rank residual speaker adapter used during target-speaker adaptation."""

    def __init__(self, hidden_dim: int = 512, rank: int = 16, alpha: int = 32, dropout: float = 0.05):
        super().__init__()
        self.scale = alpha / max(1, rank)
        self.dropout = nn.Dropout(dropout)
        self.down = nn.Linear(hidden_dim, rank, bias=False)
        self.up = nn.Linear(rank, hidden_dim, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.scale * self.up(self.dropout(self.down(hidden)))
