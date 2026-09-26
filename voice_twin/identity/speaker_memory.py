from __future__ import annotations

import numpy as np


class SpeakerMemory:
    def __init__(self, max_slots: int = 64):
        self.max_slots = max_slots
        self._items: list[np.ndarray] = []

    def add(self, embedding: np.ndarray) -> None:
        x = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(x))
        if norm > 0:
            x = x / norm
        self._items.append(x)
        if len(self._items) > self.max_slots:
            self._items.pop(0)

    def matrix(self) -> np.ndarray:
        if not self._items:
            raise ValueError("speaker memory is empty")
        return np.stack(self._items)

    def query(self, query: np.ndarray, top_k: int = 4) -> tuple[np.ndarray, np.ndarray]:
        memory = self.matrix()
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        q = q / max(float(np.linalg.norm(q)), 1e-8)
        scores = memory @ q
        idx = np.argsort(scores)[::-1][:top_k]
        return memory[idx], scores[idx]
