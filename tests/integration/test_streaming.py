from __future__ import annotations

import numpy as np

from voice_twin.acoustic.base import (
    AcousticBackend,
    GeneratedAudio,
    ProviderCapabilities,
    ProviderSynthesisRequest,
)
from voice_twin.acoustic.registry import ProviderRegistry
from voice_twin.inference.engine import VoiceTwinEngine
from voice_twin.profiles.store import VoiceProfileStore
from voice_twin.profiles.voice_dna import VoiceDNA
from voice_twin.schemas import SynthesisRequest


class NativeProvider(AcousticBackend):
    provider_name = "native"
    capabilities = ProviderCapabilities(native_voice_replication=True)

    def synthesize(self, request: ProviderSynthesisRequest) -> GeneratedAudio:
        return GeneratedAudio(
            waveform=np.zeros(4800, dtype=np.float32),
            sample_rate=24000,
            provider="native",
            native_digital_twin=True,
        )


def test_stream_is_pcm_frames_from_finalized_twin(tmp_path):
    store = VoiceProfileStore(tmp_path)
    profile = VoiceDNA(
        speaker_id="target",
        identity_core=np.ones(4, dtype=np.float32),
        vocal_profile=np.ones(3, dtype=np.float32),
        identity_memory=np.ones((2, 4), dtype=np.float32),
    )
    profile.bind_provider("native", kind="replicated", voice_id="voice_test")
    store.save(profile)

    registry = ProviderRegistry()
    registry.register(NativeProvider())
    engine = VoiceTwinEngine(registry, store, default_provider="native")

    frames = list(engine.stream(SynthesisRequest(text="hello", voice_id="target")))

    assert frames
    assert all(frame.sample_rate == 24000 for frame in frames)
    assert all(frame.format.value == "pcm_s16le" for frame in frames)
    assert sum(len(frame.pcm) for frame in frames) == 4800 * 2
