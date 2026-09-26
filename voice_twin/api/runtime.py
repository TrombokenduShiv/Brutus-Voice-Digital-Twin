from __future__ import annotations

import os
from functools import lru_cache

from voice_twin.acoustic.gemini_backend import GeminiTtsBackend
from voice_twin.acoustic.qwen_backend import QwenVoiceCloneBackend
from voice_twin.acoustic.registry import ProviderRegistry
from voice_twin.conversion.http_converter import HttpTwinConversionBackend
from voice_twin.conversion.twin_converter import DigitalTwinFinalizer
from voice_twin.inference.engine import VoiceTwinEngine
from voice_twin.profiles.store import VoiceProfileStore


@lru_cache(maxsize=1)
def get_engine() -> VoiceTwinEngine:
    registry = ProviderRegistry()
    registry.register(
        GeminiTtsBackend(
            model_id=os.getenv("BVT_GEMINI_TTS_MODEL", "gemini-3.8-flash-tts"),
            api_key=os.getenv("GEMINI_API_KEY"),
        )
    )
    registry.register(
        QwenVoiceCloneBackend(
            model_id=os.getenv("BVT_QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-Base")
        )
    )

    converter_url = os.getenv("BVT_TWIN_CONVERTER_URL")
    converter = HttpTwinConversionBackend(converter_url) if converter_url else None

    key = os.getenv("BVT_PROFILE_KEY")
    store = VoiceProfileStore(
        os.getenv("BVT_PROFILE_ROOT", "artifacts/voices"),
        encryption_key=key.encode() if key else None,
    )
    return VoiceTwinEngine(
        providers=registry,
        profiles=store,
        default_provider=os.getenv("BVT_DEFAULT_TTS_PROVIDER", "gemini"),
        finalizer=DigitalTwinFinalizer(converter=converter),
    )
