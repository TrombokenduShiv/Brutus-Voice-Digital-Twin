from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class ProviderCapabilities:
    native_voice_replication: bool = False
    style_control: bool = False
    streaming: bool = False
    reference_audio: bool = False


@dataclass(slots=True)
class ProviderSynthesisRequest:
    text: str
    language: str = "English"
    provider_voice_id: str | None = None
    provider_binding_kind: str | None = None
    style: str | None = None
    reference_audio: str | Path | tuple[np.ndarray, int] | None = None
    reference_text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GeneratedAudio:
    waveform: np.ndarray
    sample_rate: int
    provider: str
    provider_voice_id: str | None = None
    native_digital_twin: bool = False
    digital_twin: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class AcousticBackend(ABC):
    provider_name: str = "unknown"
    capabilities: ProviderCapabilities = ProviderCapabilities()

    @abstractmethod
    def synthesize(self, request: ProviderSynthesisRequest) -> GeneratedAudio:
        raise NotImplementedError
