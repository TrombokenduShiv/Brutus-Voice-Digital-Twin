from __future__ import annotations

from dataclasses import dataclass

from voice_twin.events.breath_detector import detect_breaths


@dataclass(slots=True)
class NonVerbalEvent:
    kind: str
    start_s: float
    end_s: float
    confidence: float


def detect_events(audio, sample_rate: int) -> list[NonVerbalEvent]:
    return [NonVerbalEvent("breath", e.start_s, e.end_s, e.confidence) for e in detect_breaths(audio, sample_rate)]
