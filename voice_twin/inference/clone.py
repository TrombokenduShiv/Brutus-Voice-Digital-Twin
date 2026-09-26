from __future__ import annotations

from voice_twin.acoustic.base import ProviderSynthesisRequest
from voice_twin.acoustic.qwen_backend import QwenVoiceCloneBackend


def zero_shot_clone(
    text: str,
    ref_audio,
    ref_text: str | None = None,
    language: str = "English",
):
    """Research baseline. Production output must use VoiceTwinEngine."""
    backend = QwenVoiceCloneBackend()
    return backend.synthesize(
        ProviderSynthesisRequest(
            text=text,
            language=language,
            provider_binding_kind="replicated",
            reference_audio=ref_audio,
            reference_text=ref_text,
        )
    )
