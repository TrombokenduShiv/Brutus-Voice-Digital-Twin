from __future__ import annotations

import numpy as np

from voice_twin.audio.resample import resample
from voice_twin.device.inverse_filter import apply_safe_inverse
from voice_twin.device.limiter import limiter
from voice_twin.device.transfer_function import DeviceTransfer


class RobotRenderer:
    def __init__(self, target_sr: int = 24000, device_transfer: DeviceTransfer | None = None):
        self.target_sr = target_sr
        self.device_transfer = device_transfer

    def render(self, audio: np.ndarray, source_sr: int) -> np.ndarray:
        x = resample(audio, source_sr, self.target_sr)
        if self.device_transfer is not None:
            x = apply_safe_inverse(x, self.device_transfer)
        return limiter(x)
