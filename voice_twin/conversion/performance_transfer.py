from __future__ import annotations

from dataclasses import dataclass

from voice_twin.acoustic.base import AcousticBackend, GeneratedAudio
from voice_twin.conversion.performance_features import PerformanceFeatures


@dataclass(slots=True)
class PerformanceTransferRequest:
    text: str
    language: str
    reference_features: PerformanceFeatures
    reference_audio: object
    reference_text: str | None = None


class PerformanceTransfer:
    def __init__(self, backend: AcousticBackend):
        self.backend = backend

    def synthesize(self, request: PerformanceTransferRequest) -> GeneratedAudio:
        # Foundation backend creates target identity; downstream alignment metrics verify
        # how closely explicit F0/timing controls are preserved.
        return self.backend.synthesize(
            text=request.text,
            language=request.language,
            ref_audio=request.reference_audio,
            ref_text=request.reference_text,
        )
