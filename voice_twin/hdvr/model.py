from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from training.losses.channel_adversarial import channel_confusion_loss
from training.losses.pitch import log_f0_loss
from training.losses.speaker import speaker_cosine_loss
from voice_twin.accent.accent_encoder import AccentFeatureEncoder
from voice_twin.hdvr.adapters import SpeakerLoRAAdapter
from voice_twin.hdvr.fusion import HDVRFusion


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.to(values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


class HDVRModel(nn.Module):
    """Trainable hierarchical speaker-behaviour model.

    It predicts token-level acoustics/prosody/accent/events while preserving an
    utterance-level identity representation. It does not replace the TTS provider;
    its learned representation conditions or post-converts any provider output.
    """

    def __init__(
        self,
        vocab_size: int = 2048,
        text_dim: int = 512,
        hidden_dim: int = 512,
        identity_in_dim: int = 192,
        identity_dim: int = 256,
        vocal_dim: int = 128,
        accent_feature_dim: int = 8,
        accent_dim: int = 192,
        prosody_feature_dim: int = 5,
        prosody_dim: int = 256,
        event_classes: int = 7,
        event_dim: int = 64,
        mel_bins: int = 80,
        ssl_dim: int = 768,
        channel_classes: int = 32,
        adapter_rank: int = 0,
        adapter_alpha: int = 32,
        adapter_dropout: float = 0.05,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.mel_bins = mel_bins
        self.ssl_dim = ssl_dim
        self.event_classes = event_classes

        self.phone_embedding = nn.Embedding(vocab_size, text_dim, padding_idx=0)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=text_dim,
            nhead=8,
            dim_feedforward=text_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.text_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)

        self.identity_projection = nn.Sequential(
            nn.LayerNorm(identity_in_dim),
            nn.Linear(identity_in_dim, identity_dim),
        )
        self.memory_projection = nn.Sequential(
            nn.LayerNorm(identity_in_dim),
            nn.Linear(identity_in_dim, identity_dim),
        )
        self.vocal_projection = nn.Sequential(
            nn.LayerNorm(vocal_dim),
            nn.Linear(vocal_dim, vocal_dim),
        )
        self.accent_encoder = AccentFeatureEncoder(accent_feature_dim, accent_dim)
        self.prosody_encoder = nn.Sequential(
            nn.LayerNorm(prosody_feature_dim),
            nn.Linear(prosody_feature_dim, prosody_dim),
            nn.SiLU(),
            nn.Linear(prosody_dim, prosody_dim),
        )
        self.event_encoder = nn.Sequential(
            nn.LayerNorm(event_classes),
            nn.Linear(event_classes, event_dim),
            nn.SiLU(),
        )

        self.fusion = HDVRFusion(
            text_dim=text_dim,
            identity_dim=identity_dim,
            vocal_dim=vocal_dim,
            memory_dim=identity_dim,
            accent_dim=accent_dim,
            prosody_dim=prosody_dim,
            event_dim=event_dim,
            hidden_dim=hidden_dim,
        )
        self.adapter = (
            SpeakerLoRAAdapter(hidden_dim, adapter_rank, adapter_alpha, adapter_dropout)
            if adapter_rank > 0
            else nn.Identity()
        )

        self.condition_head = nn.Linear(hidden_dim, hidden_dim)
        self.acoustic_head = nn.Linear(hidden_dim, mel_bins)
        self.prosody_head = nn.Linear(hidden_dim, 5)
        self.duration_head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Softplus())
        self.pause_type_head = nn.Linear(hidden_dim, 5)
        self.pause_duration_head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Softplus())
        self.accent_head = nn.Linear(hidden_dim, accent_feature_dim)
        self.event_head = nn.Linear(hidden_dim, event_classes)

        self.speaker_head = nn.Linear(hidden_dim, identity_in_dim)
        self.ssl_head = nn.Linear(hidden_dim, ssl_dim)
        self.channel_head = nn.Linear(hidden_dim, channel_classes)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        token_mask = batch["token_mask"].bool()
        text = self.phone_embedding(batch["token_ids"])
        text = self.text_encoder(text, src_key_padding_mask=~token_mask)

        identity = self.identity_projection(batch["identity"])
        memory = self.memory_projection(batch["speaker_memory"])
        vocal = self.vocal_projection(batch["vocal"])
        accent = self.accent_encoder(batch["accent_context"])
        prosody = self.prosody_encoder(batch["prosody_context"])
        events = self.event_encoder(batch["event_context"])

        memory_mask = torch.isfinite(batch["speaker_memory"]).all(dim=-1)
        memory = torch.nan_to_num(memory)
        hidden = self.fusion(
            text,
            identity,
            vocal,
            memory,
            accent,
            prosody,
            events,
            memory_mask=memory_mask,
        )
        hidden = self.adapter(hidden)

        mask = token_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

        raw_prosody = self.prosody_head(hidden)
        prosody_pred = torch.stack(
            [
                F.softplus(raw_prosody[..., 0]) + 1.0,
                torch.sigmoid(raw_prosody[..., 1]),
                F.softplus(raw_prosody[..., 2]),
                torch.sigmoid(raw_prosody[..., 3]),
                torch.sigmoid(raw_prosody[..., 4]),
            ],
            dim=-1,
        )
        return {
            "conditioning": self.condition_head(hidden),
            "acoustic": self.acoustic_head(hidden),
            "prosody": prosody_pred,
            "duration_ms": self.duration_head(hidden).squeeze(-1),
            "pause_type_logits": self.pause_type_head(hidden),
            "pause_duration_ms": self.pause_duration_head(hidden).squeeze(-1),
            "accent": self.accent_head(hidden),
            "event_logits": self.event_head(hidden),
            "speaker_embedding": self.speaker_head(pooled),
            "ssl_embedding": self.ssl_head(pooled),
            "channel_logits": self.channel_head(pooled),
        }

    def compute_losses(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out = self.forward(batch)
        mask = batch["token_mask"].bool()

        acoustic = _masked_mean(
            torch.abs(out["acoustic"] - batch["acoustic_target"]),
            mask,
        )
        speaker = speaker_cosine_loss(out["speaker_embedding"], batch["speaker_target"])
        pitch = log_f0_loss(
            out["prosody"][..., 0],
            batch["prosody_target"][..., 0],
            batch["prosody_target"][..., 3] > 0.5,
        )
        duration = _masked_mean(
            F.smooth_l1_loss(
                out["duration_ms"],
                batch["duration_ms"],
                reduction="none",
            ),
            mask,
        )
        pause_type = F.cross_entropy(
            out["pause_type_logits"][mask],
            batch["pause_type"][mask],
        )
        pause_duration = F.smooth_l1_loss(
            out["pause_duration_ms"][mask],
            batch["pause_duration_ms"][mask],
        )
        pause = pause_type + 0.01 * pause_duration
        accent = _masked_mean(
            F.smooth_l1_loss(
                out["accent"],
                batch["accent_target"],
                reduction="none",
            ),
            mask,
        )
        event = F.cross_entropy(
            out["event_logits"][mask],
            batch["event_target"][mask],
        )
        ssl_perceptual = F.mse_loss(out["ssl_embedding"], batch["ssl_target"])
        channel_adversarial = channel_confusion_loss(out["channel_logits"])

        return {
            "acoustic": acoustic,
            "speaker": speaker,
            "pitch": pitch,
            "duration": duration,
            "pause": pause,
            "accent": accent,
            "event": event,
            "ssl_perceptual": ssl_perceptual,
            "channel_adversarial": channel_adversarial,
        }

    def enable_speaker_adapter(self, rank: int = 16, alpha: int = 32, dropout: float = 0.05) -> None:
        hidden_dim = self.condition_head.in_features
        self.adapter = SpeakerLoRAAdapter(hidden_dim, rank, alpha, dropout).to(
            next(self.parameters()).device
        )

    def freeze_for_speaker_adaptation(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False
        for parameter in self.adapter.parameters():
            parameter.requires_grad = True
