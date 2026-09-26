from __future__ import annotations

import torch


def log_f0_loss(pred: torch.Tensor, target: torch.Tensor, voiced: torch.Tensor) -> torch.Tensor:
    p = torch.log(torch.clamp(pred, min=1.0))
    t = torch.log(torch.clamp(target, min=1.0))
    mask = voiced.to(dtype=p.dtype)
    return ((p - t).abs() * mask).sum() / mask.sum().clamp_min(1.0)
