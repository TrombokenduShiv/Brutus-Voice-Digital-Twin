from __future__ import annotations

from voice_twin.acoustic.qwen_backend import QwenVoiceCloneBackend


def zero_shot_clone(text: str, ref_audio, ref_text: str | None = None, language: str = "English"):
    backend = QwenVoiceCloneBackend()
    return backend.synthesize(text=text, language=language, ref_audio=ref_audio, ref_text=ref_text)
