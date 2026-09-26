from __future__ import annotations

import numpy as np


def f0_metrics(reference: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    n = min(len(reference), len(predicted))
    if n < 2:
        return {"f0_correlation": 0.0, "log_f0_rmse": float("inf")}
    r = np.asarray(reference[:n], dtype=np.float64)
    p = np.asarray(predicted[:n], dtype=np.float64)
    mask = (r > 0) & (p > 0)
    if mask.sum() < 2:
        return {"f0_correlation": 0.0, "log_f0_rmse": float("inf")}
    lr, lp = np.log(r[mask]), np.log(p[mask])
    return {
        "f0_correlation": float(np.corrcoef(lr, lp)[0, 1]),
        "log_f0_rmse": float(np.sqrt(np.mean((lr-lp)**2))),
    }
