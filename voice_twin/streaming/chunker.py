from __future__ import annotations

import numpy as np


def chunk_audio(audio: np.ndarray, sample_rate: int, chunk_ms: int = 60):
    x = np.asarray(audio, dtype=np.float32)
    n = max(1, round(sample_rate * chunk_ms / 1000))
    for start in range(0, len(x), n):
        yield x[start:start+n]
