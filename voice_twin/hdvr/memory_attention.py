from __future__ import annotations

import torch
from torch import nn


class MaskedMemoryAttention(nn.Module):
    def __init__(self, dim: int = 256, heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        key_padding_mask = None if memory_mask is None else ~memory_mask.bool()
        out, _ = self.attn(
            query,
            memory,
            memory,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return self.norm(query + out)
