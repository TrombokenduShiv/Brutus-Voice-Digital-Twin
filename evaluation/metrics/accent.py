from __future__ import annotations

import numpy as np

from voice_twin.accent.acoustic_features import FEATURE_NAMES, extract_accent_features


def accent_feature_distance(reference: np.ndarray, generated: np.ndarray) -> dict[str, float]:
    r = np.asarray(reference, dtype=np.float64)
    g = np.asarray(generated, dtype=np.float64)
    if r.shape != g.shape:
        n = min(len(r), len(g))
        r, g = r[:n], g[:n]
    if r.ndim == 1:
        r, g = r[None, :], g[None, :]
    scale = np.maximum(np.std(r, axis=0), 1e-6)
    z_mae = np.mean(np.abs(r - g) / scale, axis=0)
    out = {
        f"accent_mae_z_{name}": float(value)
        for name, value in zip(FEATURE_NAMES, z_mae)
    }
    out["accent_feature_distance"] = float(np.mean(z_mae))
    return out


def whole_utterance_accent_features(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    return extract_accent_features(audio, sample_rate)


def allophone_accuracy(reference_labels, generated_labels) -> float:
    r = np.asarray(reference_labels)
    g = np.asarray(generated_labels)
    n = min(len(r), len(g))
    return float(np.mean(r[:n] == g[:n])) if n else 0.0
