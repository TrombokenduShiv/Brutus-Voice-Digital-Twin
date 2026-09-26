from __future__ import annotations

import numpy as np


def channel_descriptor(audio: np.ndarray, sample_rate: int, bins: int = 32) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32)
    if not x.size:
        return np.zeros(bins, dtype=np.float32)
    spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    chunks = np.array_split(np.log1p(spectrum), bins)
    return np.asarray([float(np.mean(c)) if len(c) else 0.0 for c in chunks], dtype=np.float32)
