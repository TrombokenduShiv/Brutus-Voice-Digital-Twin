from __future__ import annotations

import numpy as np


def remove_dc(audio: np.ndarray) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32)
    return x - float(x.mean()) if x.size else x


def peak_normalize(audio: np.ndarray, target: float = 0.95) -> np.ndarray:
    x = remove_dc(audio)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak <= 1e-8:
        return x
    return (x * (target / peak)).astype(np.float32)


def to_pcm16(audio: np.ndarray) -> bytes:
    x = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    return (x * 32767.0).astype("<i2").tobytes()
