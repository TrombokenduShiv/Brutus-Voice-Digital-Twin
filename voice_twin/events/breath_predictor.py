from __future__ import annotations

import torch
from torch import nn


class BreathPredictor(nn.Module):
    def __init__(self, dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, 3))

    def forward(self, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.net(hidden)
        return {
            "probability": torch.sigmoid(out[..., 0]),
            "duration_ms": torch.nn.functional.softplus(out[..., 1]),
            "amplitude": torch.sigmoid(out[..., 2]),
        }
