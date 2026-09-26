from __future__ import annotations

import torch


def accent_feature_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.smooth_l1_loss(pred, target)
