from __future__ import annotations

import torch


def acoustic_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.l1_loss(pred, target)
