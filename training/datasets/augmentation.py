from __future__ import annotations

import torch


class FeatureAugmentor:
    """Feature-domain augmentation that preserves target pitch identity."""

    def __init__(
        self,
        probability: float = 0.35,
        context_noise_std: float = 0.02,
        memory_dropout: float = 0.10,
        prosody_mask: float = 0.10,
    ):
        self.probability = probability
        self.context_noise_std = context_noise_std
        self.memory_dropout = memory_dropout
        self.prosody_mask = prosody_mask

    def __call__(self, batch: dict) -> dict:
        if torch.rand(()) > self.probability:
            return batch
        batch = dict(batch)
        if self.context_noise_std > 0:
            batch["accent_context"] = batch["accent_context"] + (
                torch.randn_like(batch["accent_context"]) * self.context_noise_std
            )
        if self.memory_dropout > 0:
            mask = (
                torch.rand(batch["speaker_memory"].shape[:2], device=batch["speaker_memory"].device)
                > self.memory_dropout
            ).unsqueeze(-1)
            batch["speaker_memory"] = batch["speaker_memory"] * mask
        if self.prosody_mask > 0:
            mask = (
                torch.rand(batch["prosody_context"].shape[:2], device=batch["prosody_context"].device)
                > self.prosody_mask
            ).unsqueeze(-1)
            batch["prosody_context"] = batch["prosody_context"] * mask
        return batch
