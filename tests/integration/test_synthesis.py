from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from voice_twin.acoustic.base import (
    AcousticBackend,
    GeneratedAudio,
    ProviderCapabilities,
    ProviderSynthesisRequest,
)
from voice_twin.acoustic.registry import ProviderRegistry
from voice_twin.conversion.twin_converter import (
    DigitalTwinFinalizer,
    DigitalTwinInvariantError,
    TwinConversionBackend,
)
from voice_twin.inference.engine import VoiceTwinEngine
from voice_twin.profiles.store import VoiceProfileStore
from voice_twin.profiles.voice_dna import VoiceDNA
from voice_twin.schemas import SynthesisRequest


class NativeProvider(AcousticBackend):
    provider_name = "native"
    capabilities = ProviderCapabilities(native_voice_replication=True)

    def synthesize(self, request: ProviderSynthesisRequest) -> GeneratedAudio:
        assert request.provider_binding_kind == "replicated"
        return GeneratedAudio(
            waveform=np.zeros(2400, dtype=np.float32),
            sample_rate=24000,
            provider=self.provider_name,
            provider_voice_id=request.provider_voice_id,
            native_digital_twin=True,
        )


class CarrierProvider(AcousticBackend):
    provider_name = "carrier"

    def synthesize(self, request: ProviderSynthesisRequest) -> GeneratedAudio:
        return GeneratedAudio(
            waveform=np.zeros(2400, dtype=np.float32),
            sample_rate=24000,
            provider=self.provider_name,
            native_digital_twin=False,
        )


class FakeConverter(TwinConversionBackend):
    def convert(self, generated, profile, plan):
        return replace(
            generated,
            waveform=np.ones_like(generated.waveform) * 0.01,
            digital_twin=True,
            metadata={**generated.metadata, "converted_for": profile.speaker_id},
        )


def _store(tmp_path, provider: str, kind: str = "replicated") -> VoiceProfileStore:
    store = VoiceProfileStore(tmp_path)
    profile = VoiceDNA(
        speaker_id="target",
        identity_core=np.ones(4, dtype=np.float32),
        vocal_profile=np.ones(3, dtype=np.float32),
        identity_memory=np.ones((2, 4), dtype=np.float32),
    )
    profile.bind_provider(provider, kind=kind, voice_id="voice_test")
    store.save(profile)
    return store


def test_native_provider_always_leaves_as_digital_twin(tmp_path):
    registry = ProviderRegistry()
    registry.register(NativeProvider())
    engine = VoiceTwinEngine(registry, _store(tmp_path, "native"), default_provider="native")

    result = engine.synthesize(SynthesisRequest(text="hello", voice_id="target"))

    assert result.digital_twin is True
    assert result.provider == "native"
    assert result.sample_rate == 24000


def test_non_twin_provider_fails_closed_without_converter(tmp_path):
    registry = ProviderRegistry()
    registry.register(CarrierProvider())
    engine = VoiceTwinEngine(registry, _store(tmp_path, "carrier"), default_provider="carrier")

    with pytest.raises(DigitalTwinInvariantError):
        engine.synthesize(SynthesisRequest(text="hello", voice_id="target"))


def test_arbitrary_external_tts_audio_can_be_twin_converted(tmp_path):
    registry = ProviderRegistry()
    engine = VoiceTwinEngine(
        registry,
        _store(tmp_path, "anything"),
        finalizer=DigitalTwinFinalizer(FakeConverter()),
    )
    request = SynthesisRequest(text="hello", voice_id="target", provider="anything")
    carrier = np.zeros(1200, dtype=np.float32)

    result = engine.finalize_external(request, carrier, 24000, "anything")

    assert result.digital_twin is True
    assert result.metadata["converted_for"] == "target"
