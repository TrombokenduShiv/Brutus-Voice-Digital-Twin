from __future__ import annotations

import random

from torch.utils.data import Sampler


class SecondsBatchSampler(Sampler[list[int]]):
    """Variable-size batches capped by total audio seconds."""

    def __init__(
        self,
        dataset,
        max_seconds: float = 120.0,
        max_items: int = 32,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.max_seconds = max_seconds
        self.max_items = max_items
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(indices)
        batch: list[int] = []
        seconds = 0.0
        for idx in indices:
            row = self.dataset.rows[idx]
            duration = max(float(row.get("duration", 0.0)), 0.01)
            if batch and (
                seconds + duration > self.max_seconds or len(batch) >= self.max_items
            ):
                yield batch
                batch, seconds = [], 0.0
            batch.append(idx)
            seconds += duration
        if batch:
            yield batch

    def __len__(self) -> int:
        total = sum(max(float(r.get("duration", 0.0)), 0.01) for r in self.dataset.rows)
        return max(1, int(total / max(self.max_seconds, 0.01)) + 1)
