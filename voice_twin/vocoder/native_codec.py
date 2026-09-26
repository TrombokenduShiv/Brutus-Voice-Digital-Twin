from __future__ import annotations

import numpy as np

from voice_twin.vocoder.base import Vocoder


class NativeWaveformPassThrough(Vocoder):
    """Used when the acoustic backend already returns waveform audio."""

    def decode(self, features: np.ndarray) -> np.ndarray:
        return np.asarray(features, dtype=np.float32)
