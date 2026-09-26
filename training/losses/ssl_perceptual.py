from __future__ import annotations

import torch


def ssl_perceptual_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.mse_loss(pred, target)
