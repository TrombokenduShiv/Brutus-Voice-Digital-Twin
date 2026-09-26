from __future__ import annotations

import torch
from torch import nn


class PausePredictor(nn.Module):
    def __init__(self, dim: int = 256, classes: int = 5):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.type_head = nn.Linear(dim, classes)
        self.duration_head = nn.Sequential(nn.Linear(dim, 1), nn.Softplus())

    def forward(self, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.backbone(hidden)
        return {"type_logits": self.type_head(h), "duration_ms": self.duration_head(h).squeeze(-1)}
