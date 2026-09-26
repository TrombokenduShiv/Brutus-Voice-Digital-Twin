from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class BrutusAudioContract:
    sample_rate: int = 24000
    channels: int = 1
    sample_width_bits: int = 16
    encoding: str = "pcm_s16le"
    chunk_ms: int = 60
