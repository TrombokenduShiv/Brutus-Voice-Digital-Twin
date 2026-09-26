from __future__ import annotations

import librosa
import numpy as np


def extract_energy(audio: np.ndarray, sample_rate: int, frame_rate_hz: int = 100) -> np.ndarray:
    hop = max(1, round(sample_rate / frame_rate_hz))
    frame = max(256, 4 * hop)
    rms = librosa.feature.rms(y=np.asarray(audio, dtype=np.float32), frame_length=frame, hop_length=hop)[0]
    return rms.astype(np.float32)
