from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class TrainState:
    step: int = 0
    best_metric: float | None = None


class Trainer:
    def __init__(self, model, optimizer, loss_fn, clip_norm: float = 1.0):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.clip_norm = clip_norm
        self.state = TrainState()

    def train_step(self, batch) -> dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        losses = self.model.compute_losses(batch)
        total, metrics = self.loss_fn(losses)
        total.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)
        self.optimizer.step()
        self.state.step += 1
        return metrics
