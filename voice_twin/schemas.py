from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class SynthesisMode(str, Enum):
    SYNTHESIS = "synthesis"
    TWIN = "twin"
    PERFORMANCE = "performance"


class AudioFormat(str, Enum):
    PCM_S16LE = "pcm_s16le"
    WAV = "wav"


class SynthesisRequest(BaseModel):
    text: str = Field(min_length=1)
    voice_id: str
    language: str = "English"
    emotion: str | None = None
    mode: SynthesisMode = SynthesisMode.TWIN
    prosody_strength: float = Field(default=1.0, ge=0.0, le=2.0)
    accent_strength: float = Field(default=1.0, ge=0.0, le=2.0)
    reference_audio: Path | None = None
    reference_text: str | None = None
    seed: int | None = None


class AudioFrame(BaseModel):
    sequence: int
    sample_rate: int = 24_000
    channels: int = 1
    format: AudioFormat = AudioFormat.PCM_S16LE
    duration_ms: float
    pcm: bytes


class ProsodyFrame(BaseModel):
    sequence: int
    time_s: float
    f0_hz: float = 0.0
    energy: float = 0.0
    voicing: float = 0.0
    breath_probability: float = 0.0
    event: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class ConsentRecord(BaseModel):
    speaker_id: str
    statement: str
    captured_at: str
    source_sha256: str
    verified: bool = False
