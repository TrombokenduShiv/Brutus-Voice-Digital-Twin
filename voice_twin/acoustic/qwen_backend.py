from __future__ import annotations

from pathlib import Path

import numpy as np

from voice_twin.acoustic.base import AcousticBackend, GeneratedAudio


class QwenVoiceCloneBackend(AcousticBackend):
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

    def synthesize(
        self,
        text: str,
        language: str = "English",
        ref_audio: str | Path | tuple[np.ndarray, int] | None = None,
        ref_text: str | None = None,
        voice_clone_prompt=None,
        x_vector_only_mode: bool = False,
        **kwargs,
    ) -> GeneratedAudio:
        if ref_audio is None and voice_clone_prompt is None:
            raise ValueError("Qwen Base cloning requires ref_audio or a reusable voice_clone_prompt")
        model = self._load()
        wavs, sr = model.generate_voice_clone(
            text=text,
            language=language,
            ref_audio=None if voice_clone_prompt is not None else ref_audio,
            ref_text=ref_text,
            voice_clone_prompt=voice_clone_prompt,
            x_vector_only_mode=x_vector_only_mode,
            **kwargs,
        )
        return GeneratedAudio(np.asarray(wavs[0], dtype=np.float32), int(sr))

    def create_prompt(self, ref_audio, ref_text: str | None, x_vector_only_mode: bool = False):
        return self._load().create_voice_clone_prompt(
            ref_audio=ref_audio,
            ref_text=ref_text,
            x_vector_only_mode=x_vector_only_mode,
        )
