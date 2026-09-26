from __future__ import annotations

import torch


def build_optimizer(parameters, lr: float = 2e-4, weight_decay: float = 0.01):
    return torch.optim.AdamW(parameters, lr=lr, betas=(0.9, 0.98), weight_decay=weight_decay)
