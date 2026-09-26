from __future__ import annotations

from dataclasses import dataclass

from voice_twin.acoustic.base import AcousticBackend, GeneratedAudio, ProviderSynthesisRequest
from voice_twin.conversion.performance_features import PerformanceFeatures


@dataclass(slots=True)
class PerformanceTransferRequest:
    text: str
    language: str
    reference_features: PerformanceFeatures
    reference_audio: object
    reference_text: str | None = None
    provider_voice_id: str | None = None
    provider_binding_kind: str | None = None
    style: str | None = None


class PerformanceTransfer:
    """Research helper; final externally emitted audio still goes through DigitalTwinFinalizer."""

    def __init__(self, backend: AcousticBackend):
        self.backend = backend

    def synthesize(self, request: PerformanceTransferRequest) -> GeneratedAudio:
        return self.backend.synthesize(
            ProviderSynthesisRequest(
                text=request.text,
                language=request.language,
                provider_voice_id=request.provider_voice_id,
                provider_binding_kind=request.provider_binding_kind,
                style=request.style,
                reference_audio=request.reference_audio,
                reference_text=request.reference_text,
                metadata={
                    "reference_f0": request.reference_features.f0_hz,
                    "reference_energy": request.reference_features.energy,
                },
            )
        )
