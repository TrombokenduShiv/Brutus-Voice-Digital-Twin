from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class ProsodyMemoryItem:
    text_embedding: np.ndarray
    trajectory: np.ndarray
    metadata: dict


class ProsodyMemory:
    def __init__(self):
        self.items: list[ProsodyMemoryItem] = []

    def add(self, item: ProsodyMemoryItem) -> None:
        self.items.append(item)

    def retrieve(self, query: np.ndarray, k: int = 4) -> list[ProsodyMemoryItem]:
        if not self.items:
            return []
        q = np.asarray(query, dtype=np.float32)
        q /= max(float(np.linalg.norm(q)), 1e-8)
        scores = []
        for item in self.items:
            x = item.text_embedding / max(float(np.linalg.norm(item.text_embedding)), 1e-8)
            scores.append(float(x @ q))
        order = np.argsort(scores)[::-1][:k]
        return [self.items[int(i)] for i in order]
