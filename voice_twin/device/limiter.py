from __future__ import annotations

import numpy as np


def limiter(audio: np.ndarray, peak: float = 0.95) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32)
    current = float(np.max(np.abs(x))) if x.size else 0.0
    if current > peak:
        x = x * (peak / current)
    return np.clip(x, -1.0, 1.0).astype(np.float32)
