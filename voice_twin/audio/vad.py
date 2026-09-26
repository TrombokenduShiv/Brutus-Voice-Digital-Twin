from __future__ import annotations

import numpy as np


def frame_rms(audio: np.ndarray, frame_samples: int) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32)
    if x.size < frame_samples:
        x = np.pad(x, (0, frame_samples - x.size))
    n = x.size // frame_samples
    x = x[: n * frame_samples].reshape(n, frame_samples)
    return np.sqrt(np.mean(x * x, axis=1) + 1e-12)


def speech_mask(audio: np.ndarray, sample_rate: int, frame_ms: int = 20, threshold_db: float = -42.0) -> np.ndarray:
    frame = max(1, round(sample_rate * frame_ms / 1000))
    rms = frame_rms(audio, frame)
    db = 20.0 * np.log10(np.maximum(rms, 1e-8))
    return db >= threshold_db
