from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter


@dataclass
class VoiceState:
    sequence: int = 0
    request_started: float = field(default_factory=perf_counter)
    first_pcm_at: float | None = None
    generated_audio_s: float = 0.0
    generation_compute_s: float = 0.0

    def mark_pcm(self) -> None:
        if self.first_pcm_at is None:
            self.first_pcm_at = perf_counter()

    @property
    def ttfa_ms(self) -> float | None:
        return None if self.first_pcm_at is None else (self.first_pcm_at - self.request_started) * 1000.0

    @property
    def realtime_factor(self) -> float:
        return self.generation_compute_s / self.generated_audio_s if self.generated_audio_s > 0 else float("inf")
