from __future__ import annotations

import librosa
import numpy as np


def extract_f0(audio: np.ndarray, sample_rate: int, frame_rate_hz: int = 100, fmin: float = 50.0, fmax: float = 600.0) -> np.ndarray:
    hop = max(1, round(sample_rate / frame_rate_hz))
    f0, voiced_flag, _ = librosa.pyin(
        np.asarray(audio, dtype=np.float32),
        fmin=fmin,
        fmax=fmax,
        sr=sample_rate,
        hop_length=hop,
    )
    if f0 is None:
        return np.zeros(0, dtype=np.float32)
    return np.nan_to_num(f0, nan=0.0).astype(np.float32)
