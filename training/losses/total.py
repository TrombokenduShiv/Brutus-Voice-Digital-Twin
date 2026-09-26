from __future__ import annotations

from dataclasses import dataclass, field

import torch


DEFAULT_WEIGHTS = {
    "acoustic": 1.00,
    "speaker": 0.30,
    "pitch": 0.20,
    "duration": 0.15,
    "pause": 0.25,
    "accent": 0.15,
    "event": 0.10,
    "ssl_perceptual": 0.30,
    "channel_adversarial": 0.05,
}


@dataclass
class CompositeLoss:
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))

    def __call__(self, losses: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        missing = set(self.weights) - set(losses)
        if missing:
            raise KeyError(f"missing loss terms: {sorted(missing)}")
        total = sum(losses[name] * weight for name, weight in self.weights.items())
        detached = {name: float(value.detach().cpu()) for name, value in losses.items()}
        detached["total"] = float(total.detach().cpu())
        return total, detached
