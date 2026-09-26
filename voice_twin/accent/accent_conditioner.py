from __future__ import annotations

import torch
from torch import nn


class AccentConditioner(nn.Module):
    def __init__(self, phoneme_dim: int = 128, context_dim: int = 128, out_dim: int = 192):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(phoneme_dim + context_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, phoneme: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([phoneme, context], dim=-1))
