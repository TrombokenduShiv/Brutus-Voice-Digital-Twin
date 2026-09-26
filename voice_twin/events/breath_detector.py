from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class BreathEvent:
    start_s: float
    end_s: float
    confidence: float
    kind: str = "breath"


def detect_breaths(audio: np.ndarray, sample_rate: int) -> list[BreathEvent]:
    # Conservative DSP baseline: high-frequency noisy low-energy events.
    x = np.asarray(audio, dtype=np.float32)
    frame = max(1, round(sample_rate * 0.02))
    n = len(x) // frame
    events: list[BreathEvent] = []
    for i in range(n):
        chunk = x[i * frame:(i + 1) * frame]
        rms = float(np.sqrt(np.mean(chunk * chunk) + 1e-12))
        zcr = float(np.mean(np.abs(np.diff(np.signbit(chunk)))))
        if 0.002 < rms < 0.08 and zcr > 0.18:
            events.append(BreathEvent(i * 0.02, (i + 1) * 0.02, min(1.0, zcr * 2.0)))
    return events
