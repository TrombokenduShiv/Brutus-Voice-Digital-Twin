from __future__ import annotations

import numpy as np

from voice_twin.audio.normalize import to_pcm16
from voice_twin.audio.resample import resample


def pcm24k(audio: np.ndarray, sample_rate: int) -> bytes:
    return to_pcm16(resample(audio, sample_rate, 24000))
