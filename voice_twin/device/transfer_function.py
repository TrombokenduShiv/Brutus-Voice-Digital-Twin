from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class DeviceTransfer:
    frequencies_hz: np.ndarray
    response: np.ndarray
    sample_rate: int


def estimate_transfer(excitation: np.ndarray, recorded: np.ndarray, sample_rate: int, eps: float = 1e-5) -> DeviceTransfer:
    n = max(len(excitation), len(recorded))
    x = np.fft.rfft(np.pad(excitation, (0, n-len(excitation))))
    y = np.fft.rfft(np.pad(recorded, (0, n-len(recorded))))
    h = y / (x + eps)
    f = np.fft.rfftfreq(n, 1.0 / sample_rate)
    return DeviceTransfer(f.astype(np.float32), h.astype(np.complex64), sample_rate)
