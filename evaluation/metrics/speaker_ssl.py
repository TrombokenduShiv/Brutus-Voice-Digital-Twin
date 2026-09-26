from __future__ import annotations

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-12)
    return float(np.dot(a, b) / denom)


def ssl_identity_retention(
    clone_target_similarity: float,
    real_self_similarity: float,
) -> float:
    return float(clone_target_similarity / max(real_self_similarity, 1e-8))
