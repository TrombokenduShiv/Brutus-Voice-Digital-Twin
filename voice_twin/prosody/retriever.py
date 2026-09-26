from __future__ import annotations

import hashlib
import numpy as np


def deterministic_text_embedding(text: str, dim: int = 256) -> np.ndarray:
    # Stable lightweight retrieval baseline. Replace with a learned text/prosody query encoder during training.
    seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(dim).astype(np.float32)
    return x / max(float(np.linalg.norm(x)), 1e-8)
