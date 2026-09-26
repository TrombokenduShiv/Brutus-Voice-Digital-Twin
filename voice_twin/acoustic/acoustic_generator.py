from __future__ import annotations

from voice_twin.acoustic.base import AcousticBackend, GeneratedAudio


class AcousticGenerator:
    def __init__(self, backend: AcousticBackend):
        self.backend = backend

    def generate(self, text: str, language: str = "English", **conditioning) -> GeneratedAudio:
        return self.backend.synthesize(text=text, language=language, **conditioning)
