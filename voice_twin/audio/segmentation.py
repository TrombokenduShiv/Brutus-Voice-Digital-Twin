from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from voice_twin.audio.vad import speech_mask


@dataclass(slots=True)
class Segment:
    start_sample: int
    end_sample: int

    def duration_s(self, sample_rate: int) -> float:
        return (self.end_sample - self.start_sample) / sample_rate


def segment_speech(audio: np.ndarray, sample_rate: int, frame_ms: int = 20, min_ms: int = 120) -> list[Segment]:
    mask = speech_mask(audio, sample_rate, frame_ms)
    frame = max(1, round(sample_rate * frame_ms / 1000))
    segments: list[Segment] = []
    start = None
    for i, active in enumerate(mask.tolist() + [False]):
        if active and start is None:
            start = i
        elif not active and start is not None:
            if (i - start) * frame_ms >= min_ms:
                segments.append(Segment(start * frame, min(i * frame, len(audio))))
            start = None
    return segments
