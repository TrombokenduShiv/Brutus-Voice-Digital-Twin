from __future__ import annotations

import numpy as np

from voice_twin.prosody.memory import ProsodyMemory
from voice_twin.prosody.retriever import deterministic_text_embedding


class ProsodyPlanner:
    def __init__(self, memory: ProsodyMemory):
        self.memory = memory

    def retrieve_context(self, text: str, k: int = 4) -> list[np.ndarray]:
        q = deterministic_text_embedding(text)
        return [item.trajectory for item in self.memory.retrieve(q, k=k)]

    def prior(self, text: str, frames: int, features: int = 5) -> np.ndarray:
        contexts = self.retrieve_context(text)
        if not contexts:
            return np.zeros((frames, features), dtype=np.float32)
        aligned = []
        for c in contexts:
            idx = np.linspace(0, max(0, len(c) - 1), frames).astype(int)
            aligned.append(c[idx, :features])
        return np.mean(np.stack(aligned), axis=0).astype(np.float32)
