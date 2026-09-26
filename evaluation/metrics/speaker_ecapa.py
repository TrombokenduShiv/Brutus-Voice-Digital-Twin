from __future__ import annotations

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a=np.asarray(a,dtype=float).reshape(-1); b=np.asarray(b,dtype=float).reshape(-1)
    return float(a@b/(max(np.linalg.norm(a),1e-8)*max(np.linalg.norm(b),1e-8)))


def identity_retention(clone_target_similarity: float, real_self_similarity: float) -> float:
    return float(clone_target_similarity / max(real_self_similarity, 1e-8))
