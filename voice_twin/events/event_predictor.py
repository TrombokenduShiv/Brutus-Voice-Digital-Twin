from __future__ import annotations

import torch
from torch import nn


class EventPredictor(nn.Module):
    def __init__(self, dim: int = 256, num_classes: int = 7):
        super().__init__()
        self.classifier = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, num_classes))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.classifier(hidden)
