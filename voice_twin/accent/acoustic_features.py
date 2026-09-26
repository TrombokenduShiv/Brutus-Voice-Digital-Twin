from __future__ import annotations

import numpy as np
import librosa

from voice_twin.prosody.f0 import extract_f0


FEATURE_NAMES = (
    "duration_ms",
    "f0_mean_hz",
    "f0_delta_hz",
    "energy_rms",
    "spectral_centroid_hz",
    "zero_crossing_rate",
    "formant_f1_hz",
    "formant_f2_hz",
)


def _formants(segment: np.ndarray, sample_rate: int) -> tuple[float, float]:
    x = np.asarray(segment, dtype=np.float64)
    if len(x) < max(64, sample_rate // 50):
        return 0.0, 0.0
    x = x - x.mean()
    x = np.append(x[0], x[1:] - 0.97 * x[:-1])
    try:
        order = min(24, max(8, sample_rate // 1000 + 2))
        a = librosa.lpc(x, order=order)
        roots = np.roots(a)
        roots = roots[np.imag(roots) >= 0]
        angles = np.arctan2(np.imag(roots), np.real(roots))
        freqs = np.sort(angles * sample_rate / (2 * np.pi))
        freqs = freqs[(freqs > 90) & (freqs < 5000)]
        if len(freqs) >= 2:
            return float(freqs[0]), float(freqs[1])
    except Exception:
        pass
    return 0.0, 0.0


def extract_accent_features(segment: np.ndarray, sample_rate: int) -> np.ndarray:
    x = np.asarray(segment, dtype=np.float32)
    if not len(x):
        return np.zeros(len(FEATURE_NAMES), dtype=np.float32)
    f0 = extract_f0(x, sample_rate)
    voiced = f0[f0 > 0]
    f0_mean = float(voiced.mean()) if len(voiced) else 0.0
    f0_delta = float(voiced[-1] - voiced[0]) if len(voiced) > 1 else 0.0
    energy = float(np.sqrt(np.mean(x * x) + 1e-12))
    centroid = float(librosa.feature.spectral_centroid(y=x, sr=sample_rate).mean())
    zcr = float(librosa.feature.zero_crossing_rate(x).mean())
    f1, f2 = _formants(x, sample_rate)
    return np.asarray(
        [
            len(x) * 1000.0 / sample_rate,
            f0_mean,
            f0_delta,
            energy,
            centroid,
            zcr,
            f1,
            f2,
        ],
        dtype=np.float32,
    )
