from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class AudioQuality:
    duration_s: float
    rms: float
    peak: float
    clipping_fraction: float
    silence_fraction: float
    score: float


def analyze_quality(audio: np.ndarray, sample_rate: int, silence_threshold: float = 0.003) -> AudioQuality:
    x = np.asarray(audio, dtype=np.float32)
    if x.size == 0:
        return AudioQuality(0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    rms = float(np.sqrt(np.mean(x * x) + 1e-12))
    peak = float(np.max(np.abs(x)))
    clipping = float(np.mean(np.abs(x) >= 0.999))
    silence = float(np.mean(np.abs(x) < silence_threshold))
    score = float(np.clip(1.0 - clipping * 20.0 - max(0.0, silence - 0.65), 0.0, 1.0))
    return AudioQuality(x.size / sample_rate, rms, peak, clipping, silence, score)
