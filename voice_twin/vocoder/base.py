from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np


class Vocoder(ABC):
    @abstractmethod
    def decode(self, features: np.ndarray) -> np.ndarray:
        raise NotImplementedError
