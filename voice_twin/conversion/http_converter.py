from __future__ import annotations

import base64
import json
from urllib.request import Request, urlopen

import numpy as np

from voice_twin.acoustic.base import GeneratedAudio
from voice_twin.audio.normalize import to_pcm16
from voice_twin.conversion.twin_converter import TwinConversionBackend
from voice_twin.hdvr.conditioning import DigitalTwinPlan
from voice_twin.profiles.voice_dna import VoiceDNA


class HttpTwinConversionBackend(TwinConversionBackend):
    """Provider-independent adapter for a trained neural voice-conversion service."""

    def __init__(self, endpoint: str, timeout_s: float = 30.0):
        self.endpoint = endpoint
        self.timeout_s = timeout_s

    def convert(
        self,
        generated: GeneratedAudio,
        profile: VoiceDNA,
        plan: DigitalTwinPlan,
    ) -> GeneratedAudio:
        body = json.dumps({
            "source": {
                "provider": generated.provider,
                "sample_rate": generated.sample_rate,
                "pcm_s16le_base64": base64.b64encode(to_pcm16(generated.waveform)).decode("ascii"),
            },
            "voice_dna": profile.to_jsonable(),
            "plan": {
                "plan_id": plan.plan_id,
                "style_instruction": plan.style_instruction,
                "metadata": plan.metadata,
            },
        }).encode("utf-8")

        req = Request(
            self.endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=self.timeout_s) as response:
            payload = json.loads(response.read())

        raw = base64.b64decode(payload["pcm_s16le_base64"])
        waveform = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        return GeneratedAudio(
            waveform=waveform.copy(),
            sample_rate=int(payload.get("sample_rate", 24000)),
            provider=generated.provider,
            provider_voice_id=generated.provider_voice_id,
            native_digital_twin=False,
            digital_twin=True,
            metadata={
                **generated.metadata,
                "conversion_backend": "http",
                "source_provider": generated.provider,
            },
        )
