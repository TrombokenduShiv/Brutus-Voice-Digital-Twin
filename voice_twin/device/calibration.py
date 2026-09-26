from __future__ import annotations

import numpy as np


def logarithmic_sweep(sample_rate: int = 24000, seconds: float = 5.0, f0: float = 60.0, f1: float = 10000.0) -> np.ndarray:
    t = np.linspace(0, seconds, int(sample_rate * seconds), endpoint=False)
    k = np.log(f1 / f0) / seconds
    phase = 2 * np.pi * f0 * (np.exp(k * t) - 1) / k
    return (0.25 * np.sin(phase)).astype(np.float32)
