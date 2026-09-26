from __future__ import annotations

import numpy as np


def duration_mae_ms(reference_ms, predicted_ms) -> float:
    r, p = np.asarray(reference_ms, dtype=float), np.asarray(predicted_ms, dtype=float)
    n = min(len(r), len(p))
    return float(np.mean(np.abs(r[:n]-p[:n]))) if n else 0.0
