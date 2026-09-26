from __future__ import annotations

import torch


def pause_loss(type_logits: torch.Tensor, type_target: torch.Tensor, duration_pred: torch.Tensor, duration_target: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cross_entropy(type_logits, type_target) + 0.01 * torch.nn.functional.smooth_l1_loss(duration_pred, duration_target)
