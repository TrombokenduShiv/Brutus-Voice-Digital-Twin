from __future__ import annotations

import torch
from torch import nn

from voice_twin.hdvr.fusion import HDVRFusion


class HDVRModel(nn.Module):
    """Conditioning network. Acoustic foundation-model integration is intentionally separate."""

    def __init__(self, text_dim: int = 512, hidden_dim: int = 512):
        super().__init__()
        self.fusion = HDVRFusion(text_dim=text_dim, hidden_dim=hidden_dim)
        self.prosody_head = nn.Linear(hidden_dim, 5)
        self.condition_head = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, text, identity, vocal, memory, accent, prosody, events):
        hidden = self.fusion(text, identity, vocal, memory, accent, prosody, events)
        return {
            "conditioning": self.condition_head(hidden),
            "prosody_prediction": self.prosody_head(hidden),
        }
