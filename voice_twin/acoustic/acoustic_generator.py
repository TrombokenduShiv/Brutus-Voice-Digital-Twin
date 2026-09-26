from __future__ import annotations

from voice_twin.acoustic.base import AcousticBackend, GeneratedAudio, ProviderSynthesisRequest


class AcousticGenerator:
    """Internal provider adapter. External callers must use VoiceTwinEngine."""

    def __init__(self, backend: AcousticBackend):
        self.backend = backend

    def generate(self, request: ProviderSynthesisRequest) -> GeneratedAudio:
        return self.backend.synthesize(request)
