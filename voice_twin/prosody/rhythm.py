from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class RhythmProfile:
    syllables_per_second: float
    pause_fraction: float
    mean_pause_ms: float
    phrase_rate: float
