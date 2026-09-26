from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class GeneratedAudio:
    waveform: np.ndarray
    sample_rate: int


class AcousticBackend(ABC):
    @abstractmethod
    def synthesize(self, text: str, language: str, **kwargs) -> GeneratedAudio:
        raise NotImplementedError
