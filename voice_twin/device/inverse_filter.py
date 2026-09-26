from __future__ import annotations

import numpy as np

from voice_twin.device.transfer_function import DeviceTransfer


def apply_safe_inverse(audio: np.ndarray, transfer: DeviceTransfer, max_gain_db: float = 9.0) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32)
    n = len(x)
    spec = np.fft.rfft(x)
    target_f = np.fft.rfftfreq(n, 1.0 / transfer.sample_rate)
    mag = np.interp(target_f, transfer.frequencies_hz, np.abs(transfer.response))
    max_gain = 10 ** (max_gain_db / 20)
    inv = np.clip(1.0 / np.maximum(mag, 1e-3), 1.0 / max_gain, max_gain)
    out = np.fft.irfft(spec * inv, n=n)
    return out.astype(np.float32)
