from __future__ import annotations

import torch


def duration_loss(pred_ms: torch.Tensor, target_ms: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.smooth_l1_loss(pred_ms, target_ms)
