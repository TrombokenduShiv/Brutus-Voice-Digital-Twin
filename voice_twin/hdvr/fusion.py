from __future__ import annotations

import torch
from torch import nn

from voice_twin.identity.multi_reference_attention import MultiReferenceAttention


class HDVRFusion(nn.Module):
    def __init__(
        self,
        text_dim: int = 512,
        identity_dim: int = 256,
        vocal_dim: int = 128,
        memory_dim: int = 256,
        accent_dim: int = 192,
        prosody_dim: int = 256,
        event_dim: int = 64,
        hidden_dim: int = 512,
        heads: int = 8,
    ):
        super().__init__()
        self.memory_query = nn.Linear(text_dim, memory_dim)
        self.memory_attn = MultiReferenceAttention(memory_dim, heads)
        in_dim = text_dim + identity_dim + vocal_dim + memory_dim + accent_dim + prosody_dim + event_dim
        self.project = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, text, identity, vocal, memory, accent, prosody, events):
        q = self.memory_query(text)
        attended = self.memory_attn(q, memory)
        static_identity = identity.unsqueeze(1).expand(-1, text.shape[1], -1)
        vocal = vocal.unsqueeze(1).expand(-1, text.shape[1], -1)
        return self.project(torch.cat([text, static_identity, vocal, attended, accent, prosody, events], dim=-1))
