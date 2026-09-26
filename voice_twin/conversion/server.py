from __future__ import annotations

import base64
import os
from functools import lru_cache

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

from voice_twin.acoustic.base import GeneratedAudio
from voice_twin.audio.normalize import to_pcm16
from voice_twin.conversion.local_neural_converter import LocalNeuralTwinConverter
from voice_twin.hdvr.conditioning import DigitalTwinPlan
from voice_twin.profiles.voice_dna import VoiceDNA


class SourcePayload(BaseModel):
    provider: str
    sample_rate: int
    pcm_s16le_base64: str


class ConvertRequest(BaseModel):
    source: SourcePayload
    voice_dna: dict
    plan: dict


@lru_cache(maxsize=1)
def converter() -> LocalNeuralTwinConverter:
    checkpoint = os.environ["BVT_TWIN_CONVERTER_CHECKPOINT"]
    return LocalNeuralTwinConverter(
        checkpoint,
        device=os.getenv("BVT_CONVERTER_DEVICE", "cuda"),
        vocoder_model=os.getenv(
            "BVT_BIGVGAN_MODEL",
            "nvidia/bigvgan_v2_24khz_100band_256x",
        ),
        use_cuda_kernel=os.getenv("BVT_BIGVGAN_CUDA_KERNEL", "false").lower() == "true",
    )


app = FastAPI(title="BRUTUS Neural Twin Converter", version="0.1.0")


@app.get("/health")
def health():
    return {"ok": True, "service": "brutus-neural-twin-converter"}


@app.post("/v1/convert")
def convert(request: ConvertRequest):
    raw = base64.b64decode(request.source.pcm_s16le_base64)
    waveform = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    source = GeneratedAudio(
        waveform=waveform.copy(),
        sample_rate=request.source.sample_rate,
        provider=request.source.provider,
    )
    profile = VoiceDNA.from_jsonable(request.voice_dna)
    plan = DigitalTwinPlan(
        plan_id=str(request.plan.get("plan_id", "external")),
        provider=request.source.provider,
        provider_voice_id=None,
        provider_binding_kind=None,
        style_instruction=str(request.plan.get("style_instruction", "")),
        metadata=dict(request.plan.get("metadata", {})),
    )
    output = converter().convert(source, profile, plan)
    return {
        "sample_rate": output.sample_rate,
        "pcm_s16le_base64": base64.b64encode(to_pcm16(output.waveform)).decode("ascii"),
        "digital_twin": True,
    }
