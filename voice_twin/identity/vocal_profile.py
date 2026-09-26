from __future__ import annotations

import numpy as np


def vocal_profile(f0_hz: np.ndarray, energy: np.ndarray, dim: int = 128) -> np.ndarray:
    f0 = np.asarray(f0_hz, dtype=np.float32)
    e = np.asarray(energy, dtype=np.float32)
    voiced = f0[f0 > 0]
    stats = np.array([
        float(np.mean(voiced)) if voiced.size else 0.0,
        float(np.std(voiced)) if voiced.size else 0.0,
        float(np.percentile(voiced, 10)) if voiced.size else 0.0,
        float(np.percentile(voiced, 90)) if voiced.size else 0.0,
        float(np.mean(e)) if e.size else 0.0,
        float(np.std(e)) if e.size else 0.0,
    ], dtype=np.float32)
    return np.pad(stats, (0, max(0, dim - len(stats))))[:dim]
