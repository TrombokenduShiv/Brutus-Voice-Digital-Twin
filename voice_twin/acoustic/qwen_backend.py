from __future__ import annotations

import numpy as np

from voice_twin.acoustic.base import (
    AcousticBackend,
    GeneratedAudio,
    ProviderCapabilities,
    ProviderSynthesisRequest,
)


class QwenVoiceCloneBackend(AcousticBackend):
    provider_name = "qwen3_tts"
    capabilities = ProviderCapabilities(
        native_voice_replication=True,
        style_control=False,
        streaming=False,
        reference_audio=True,
    )

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        device_map: str = "auto",
        dtype: str = "bfloat16",
        attn_implementation: str | None = None,
    ):
        self.model_id = model_id
        self.device_map = device_map
        self.dtype = dtype
        self.attn_implementation = attn_implementation
        self._model = None

    def _load(self):
        if self._model is None:
            import torch
            from qwen_tts import Qwen3TTSModel

            kwargs = {"device_map": self.device_map, "dtype": getattr(torch, self.dtype)}
            if self.attn_implementation:
                kwargs["attn_implementation"] = self.attn_implementation
            self._model = Qwen3TTSModel.from_pretrained(self.model_id, **kwargs)
        return self._model

    def synthesize(self, request: ProviderSynthesisRequest) -> GeneratedAudio:
        prompt = request.metadata.get("voice_clone_prompt")
        if request.reference_audio is None and prompt is None:
            raise ValueError("Qwen cloning requires target reference audio or a reusable clone prompt")
        model = self._load()
        wavs, sr = model.generate_voice_clone(
            text=request.text,
            language=request.language,
            ref_audio=None if prompt is not None else request.reference_audio,
            ref_text=request.reference_text,
            voice_clone_prompt=prompt,
            x_vector_only_mode=bool(request.metadata.get("x_vector_only_mode", False)),
        )
        return GeneratedAudio(
            waveform=np.asarray(wavs[0], dtype=np.float32),
            sample_rate=int(sr),
            provider=self.provider_name,
            provider_voice_id=request.provider_voice_id,
            native_digital_twin=True,
            metadata={"model_id": self.model_id},
        )

    def create_prompt(self, ref_audio, ref_text: str | None, x_vector_only_mode: bool = False):
        return self._load().create_voice_clone_prompt(
            ref_audio=ref_audio,
            ref_text=ref_text,
            x_vector_only_mode=x_vector_only_mode,
        )
