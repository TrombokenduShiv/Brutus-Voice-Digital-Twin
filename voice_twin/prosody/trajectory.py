from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class ProsodyTrajectory:
    times_s: np.ndarray
    f0_hz: np.ndarray
    energy: np.ndarray
    duration_rate: np.ndarray
    voicing: np.ndarray
    breath_probability: np.ndarray

    def validate(self) -> None:
        lengths = {len(self.times_s), len(self.f0_hz), len(self.energy), len(self.duration_rate), len(self.voicing), len(self.breath_probability)}
        if len(lengths) != 1:
            raise ValueError("all prosody tracks must have equal length")

    def matrix(self) -> np.ndarray:
        self.validate()
        log_f0 = np.log(np.maximum(self.f0_hz, 1.0))
        log_f0 = np.where(self.f0_hz > 0, log_f0, 0.0)
        return np.stack([log_f0, self.energy, self.duration_rate, self.voicing, self.breath_probability], axis=-1).astype(np.float32)
