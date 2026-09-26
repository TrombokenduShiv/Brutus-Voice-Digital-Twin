from __future__ import annotations

import librosa
import numpy as np


def resample(audio: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32)
    if source_sr == target_sr:
        return x
    return librosa.resample(x, orig_sr=source_sr, target_sr=target_sr, res_type="soxr_hq").astype(np.float32)
