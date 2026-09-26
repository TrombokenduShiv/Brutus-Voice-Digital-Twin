from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import replace

from voice_twin.acoustic.base import GeneratedAudio
from voice_twin.hdvr.conditioning import DigitalTwinPlan
from voice_twin.profiles.voice_dna import VoiceDNA


class DigitalTwinInvariantError(RuntimeError):
    pass


class TwinConversionBackend(ABC):
    @abstractmethod
    def convert(
        self,
        generated: GeneratedAudio,
        profile: VoiceDNA,
        plan: DigitalTwinPlan,
    ) -> GeneratedAudio:
        raise NotImplementedError


class DigitalTwinFinalizer:
    """The only legal gateway from provider audio to externally emitted audio."""

    def __init__(self, converter: TwinConversionBackend | None = None):
        self.converter = converter

    def finalize(
        self,
        generated: GeneratedAudio,
        profile: VoiceDNA,
        plan: DigitalTwinPlan,
    ) -> GeneratedAudio:
        if generated.native_digital_twin:
            final = replace(generated, digital_twin=True)
        elif self.converter is not None:
            final = self.converter.convert(generated, profile, plan)
        else:
            raise DigitalTwinInvariantError(
                f"provider '{generated.provider}' emitted carrier audio without target-voice "
                "replication and no digital-twin conversion backend is configured"
            )

        if not final.digital_twin:
            raise DigitalTwinInvariantError("converter returned audio not marked as digital twin")
        final.metadata["voice_id"] = profile.speaker_id
        final.metadata["digital_twin_plan"] = plan.plan_id
        return final
