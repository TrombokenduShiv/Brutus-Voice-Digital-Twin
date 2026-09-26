from __future__ import annotations

import numpy as np


def time_warp(source: np.ndarray, target_samples: int) -> np.ndarray:
    x = np.asarray(source, dtype=np.float32)
    if target_samples <= 0:
        return np.zeros(0, dtype=np.float32)
    if not len(x):
        return np.zeros(target_samples, dtype=np.float32)
    old = np.linspace(0.0, 1.0, len(x), endpoint=True)
    new = np.linspace(0.0, 1.0, target_samples, endpoint=True)
    return np.interp(new, old, x).astype(np.float32)
