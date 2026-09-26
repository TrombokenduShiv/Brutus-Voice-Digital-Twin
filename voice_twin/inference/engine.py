from __future__ import annotations

from dataclasses import replace

from voice_twin.acoustic.base import GeneratedAudio, ProviderSynthesisRequest
from voice_twin.acoustic.registry import ProviderRegistry
from voice_twin.conversion.twin_converter import DigitalTwinFinalizer
from voice_twin.device.robot_renderer import RobotRenderer
from voice_twin.hdvr.conditioning import build_digital_twin_plan
from voice_twin.profiles.store import VoiceProfileStore
from voice_twin.schemas import SynthesisRequest
from voice_twin.streaming.pcm_stream import audio_frames


class VoiceTwinEngine:
    """Fail-closed orchestration: provider audio can never leave without twin finalization."""

    def __init__(
        self,
        providers: ProviderRegistry,
        profiles: VoiceProfileStore,
        default_provider: str = "gemini",
        finalizer: DigitalTwinFinalizer | None = None,
        renderer: RobotRenderer | None = None,
    ):
        self.providers = providers
        self.profiles = profiles
        self.default_provider = default_provider
        self.finalizer = finalizer or DigitalTwinFinalizer()
        self.renderer = renderer or RobotRenderer()

    def synthesize(self, request: SynthesisRequest) -> GeneratedAudio:
        profile = self.profiles.load(request.voice_id)
        provider_name = request.provider or self.default_provider
        provider = self.providers.get(provider_name)
        plan = build_digital_twin_plan(profile, request, provider_name)
        binding = plan.metadata.get("binding", {})

        provider_request = ProviderSynthesisRequest(
            text=request.text,
            language=request.language,
            provider_voice_id=plan.provider_voice_id,
            provider_binding_kind=plan.provider_binding_kind,
            style=plan.style_instruction,
            reference_audio=request.reference_audio or binding.get("reference_audio"),
            reference_text=request.reference_text or binding.get("reference_text"),
            metadata={
                "voice_clone_prompt": binding.get("voice_clone_prompt"),
                "x_vector_only_mode": binding.get("x_vector_only_mode", False),
                "plan_id": plan.plan_id,
            },
        )
        source = provider.synthesize(provider_request)
        twin = self.finalizer.finalize(source, profile, plan)

        rendered = self.renderer.render(twin.waveform, twin.sample_rate)
        final = replace(twin, waveform=rendered, sample_rate=self.renderer.target_sr)
        if not final.digital_twin:
            raise RuntimeError("digital-twin postcondition violated")
        return final

    def stream(self, request: SynthesisRequest):
        generated = self.synthesize(request)
        return audio_frames(generated.waveform, generated.sample_rate)
