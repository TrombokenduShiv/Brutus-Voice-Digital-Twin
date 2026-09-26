from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _masked(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    expanded = mask.expand_as(values).to(values.dtype)
    return (values * expanded).sum() / expanded.sum().clamp_min(1.0)


class TwinMelConverter(nn.Module):
    """Streaming-friendly speaker-conditioned mel-to-mel Digital Twin converter."""

    def __init__(
        self,
        mel_bins: int = 100,
        hidden_dim: int = 256,
        identity_dim: int = 192,
        vocal_dim: int = 128,
        style_dim: int = 20,
        layers: int = 6,
        heads: int = 8,
    ):
        super().__init__()
        self.model_config = {
            "mel_bins": mel_bins,
            "hidden_dim": hidden_dim,
            "identity_dim": identity_dim,
            "vocal_dim": vocal_dim,
            "style_dim": style_dim,
            "layers": layers,
            "heads": heads,
        }
        self.source_projection = nn.Sequential(
            nn.LayerNorm(mel_bins),
            nn.Linear(mel_bins, hidden_dim),
            nn.SiLU(),
        )
        condition_dim = identity_dim + vocal_dim + style_dim
        self.condition = nn.Sequential(
            nn.LayerNorm(condition_dim),
            nn.Linear(condition_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.sequence_model = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.mel_head = nn.Linear(hidden_dim, mel_bins)
        self.identity_head = nn.Linear(hidden_dim, identity_dim)

    def forward(
        self,
        source_mel: torch.Tensor,
        identity: torch.Tensor,
        vocal: torch.Tensor,
        style: torch.Tensor,
        frame_mask: torch.Tensor | None = None,
        *,
        causal: bool = True,
    ) -> dict[str, torch.Tensor]:
        hidden = self.source_projection(source_mel)
        cond = self.condition(torch.cat([identity, vocal, style], dim=-1))
        gamma, beta = cond.chunk(2, dim=-1)
        hidden = hidden * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)

        seq_len = hidden.shape[1]
        causal_mask = (
            torch.triu(
                torch.ones(seq_len, seq_len, device=hidden.device, dtype=torch.bool),
                diagonal=1,
            )
            if causal
            else None
        )
        hidden = self.sequence_model(
            hidden,
            mask=causal_mask,
            src_key_padding_mask=None if frame_mask is None else ~frame_mask.bool(),
        )
        hidden = self.output_norm(hidden)
        if frame_mask is None:
            pooled = hidden.mean(dim=1)
        else:
            m = frame_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
        return {
            "mel": self.mel_head(hidden),
            "identity_embedding": self.identity_head(pooled),
        }

    def compute_losses(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out = self(
            batch["source_mel"],
            batch["identity"],
            batch["vocal"],
            batch["style"],
            batch["frame_mask"],
            causal=True,
        )
        mask = batch["frame_mask"]
        mel = _masked(torch.abs(out["mel"] - batch["target_mel"]), mask)
        pred_delta = out["mel"][:, 1:] - out["mel"][:, :-1]
        target_delta = batch["target_mel"][:, 1:] - batch["target_mel"][:, :-1]
        delta_mask = mask[:, 1:] & mask[:, :-1]
        delta = _masked(torch.abs(pred_delta - target_delta), delta_mask)
        identity = (
            1.0
            - F.cosine_similarity(
                out["identity_embedding"],
                batch["identity"],
                dim=-1,
            )
        ).mean()
        return {
            "conversion_mel": mel,
            "conversion_delta": delta,
            "conversion_identity": identity,
        }
