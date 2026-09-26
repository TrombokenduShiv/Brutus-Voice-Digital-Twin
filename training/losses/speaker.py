from __future__ import annotations

import torch
import torch.nn.functional as F


def speaker_cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (1.0 - F.cosine_similarity(pred, target, dim=-1)).mean()
