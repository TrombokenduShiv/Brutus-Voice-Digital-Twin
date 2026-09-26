from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class Pause:
    start_s: float
    end_s: float
    duration_s: float
    kind: str = "silence"


def detect_pauses(audio: np.ndarray, sample_rate: int, frame_ms: int = 10, threshold_db: float = -45.0, min_ms: int = 60) -> list[Pause]:
    frame = max(1, round(sample_rate * frame_ms / 1000))
    x = np.asarray(audio, dtype=np.float32)
    n = max(1, int(np.ceil(len(x) / frame)))
    padded = np.pad(x, (0, n * frame - len(x)))
    rms = np.sqrt(np.mean(padded.reshape(n, frame) ** 2, axis=1) + 1e-12)
    silent = 20 * np.log10(np.maximum(rms, 1e-8)) < threshold_db
    result: list[Pause] = []
    start = None
    for i, flag in enumerate(silent.tolist() + [False]):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            duration_ms = (i - start) * frame_ms
            if duration_ms >= min_ms:
                s = start * frame_ms / 1000
                e = i * frame_ms / 1000
                result.append(Pause(s, e, e - s))
            start = None
    return result
