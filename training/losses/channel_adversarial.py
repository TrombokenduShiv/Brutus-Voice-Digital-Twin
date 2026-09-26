from __future__ import annotations

import torch


def channel_confusion_loss(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    uniform = torch.full_like(probs, 1.0 / probs.shape[-1])
    return torch.nn.functional.kl_div(torch.log(probs.clamp_min(1e-8)), uniform, reduction="batchmean")
