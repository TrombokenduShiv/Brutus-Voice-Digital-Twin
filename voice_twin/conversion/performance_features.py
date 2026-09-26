from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from voice_twin.events.breath_detector import BreathEvent, detect_breaths
from voice_twin.prosody.energy import extract_energy
from voice_twin.prosody.f0 import extract_f0
from voice_twin.prosody.pause_detector import Pause, detect_pauses


@dataclass(slots=True)
class PerformanceFeatures:
    f0_hz: np.ndarray
    energy: np.ndarray
    pauses: list[Pause]
    breaths: list[BreathEvent]


def extract_performance(audio: np.ndarray, sample_rate: int) -> PerformanceFeatures:
    return PerformanceFeatures(
        f0_hz=extract_f0(audio, sample_rate),
        energy=extract_energy(audio, sample_rate),
        pauses=detect_pauses(audio, sample_rate),
        breaths=detect_breaths(audio, sample_rate),
    )
