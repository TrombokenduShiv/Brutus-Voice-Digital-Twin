from __future__ import annotations

import numpy as np


def aggregate_identity(embeddings: np.ndarray) -> np.ndarray:
    x = np.asarray(embeddings, dtype=np.float32)
    if x.ndim != 2 or not len(x):
        raise ValueError("expected embeddings [n, d]")
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
    center = np.median(x, axis=0)
    return center / max(float(np.linalg.norm(center)), 1e-8)
