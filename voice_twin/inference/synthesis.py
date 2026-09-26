from __future__ import annotations

from voice_twin.inference.engine import VoiceTwinEngine
from voice_twin.schemas import SynthesisRequest


def synthesize(engine: VoiceTwinEngine, request: SynthesisRequest, **kwargs):
    return engine.synthesize(request, **kwargs)
