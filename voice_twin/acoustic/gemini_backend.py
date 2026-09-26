from __future__ import annotations

import base64
import os

import numpy as np

from voice_twin.acoustic.base import (
    AcousticBackend,
    GeneratedAudio,
    ProviderCapabilities,
    ProviderSynthesisRequest,
)


def _decode_l16(data: str | bytes) -> np.ndarray:
    raw = base64.b64decode(data) if isinstance(data, str) else bytes(data)
    return (np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0).copy()


class GeminiTtsBackend(AcousticBackend):
    provider_name = "gemini"
    capabilities = ProviderCapabilities(
        native_voice_replication=True,
        style_control=True,
        streaming=True,
        reference_audio=False,
    )

    def __init__(
        self,
        model_id: str = "gemini-3.8-flash-tts",
        api_key: str | None = None,
        sample_rate: int = 24000,
    ):
        self.model_id = model_id
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.sample_rate = sample_rate
        self._client = None

    def _load(self):
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=self.api_key) if self.api_key else genai.Client()
        return self._client

    def synthesize(self, request: ProviderSynthesisRequest) -> GeneratedAudio:
        if request.provider_binding_kind != "replicated":
            raise ValueError(
                "Gemini digital-twin synthesis requires a consent-backed replicated voice binding"
            )
        if not request.provider_voice_id:
            raise ValueError("Gemini replicated voice_id/voicekey is missing")

        annotations = []
        if request.style:
            annotations.append({"type": "speech_metadata", "style": request.style})

        interaction = self._load().interactions.create(
            model=self.model_id,
            input=[{
                "type": "user_input",
                "content": [{
                    "type": "text",
                    "text": request.text,
                    "annotations": annotations,
                }],
            }],
            response_format={
                "type": "audio",
                "mime_type": "audio/l16",
                "sample_rate": self.sample_rate,
            },
            generation_config={
                "speech_config": [{"voice": request.provider_voice_id}],
            },
        )
        if interaction.output_audio is None or interaction.output_audio.data is None:
            raise RuntimeError("Gemini returned no audio payload")

        return GeneratedAudio(
            waveform=_decode_l16(interaction.output_audio.data),
            sample_rate=self.sample_rate,
            provider=self.provider_name,
            provider_voice_id=request.provider_voice_id,
            native_digital_twin=True,
            metadata={"model_id": self.model_id, "binding_kind": "replicated"},
        )
