from __future__ import annotations

import base64

import numpy as np
from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel, Field

from voice_twin.api.runtime import get_engine
from voice_twin.audio.normalize import to_pcm16
from voice_twin.conversion.twin_converter import DigitalTwinInvariantError
from voice_twin.schemas import SynthesisRequest

router = APIRouter(tags=["voice"])


def _audio_response(generated, voice_id: str) -> Response:
    return Response(
        content=to_pcm16(generated.waveform),
        media_type="audio/l16",
        headers={
            "X-BVT-Digital-Twin": "true",
            "X-BVT-Provider": generated.provider,
            "X-BVT-Voice-Id": voice_id,
            "X-BVT-Sample-Rate": str(generated.sample_rate),
        },
    )


@router.post("/synthesize")
def synthesize(request: SynthesisRequest) -> Response:
    try:
        return _audio_response(get_engine().synthesize(request), request.voice_id)
    except (FileNotFoundError, KeyError, ValueError, DigitalTwinInvariantError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"TTS runtime unavailable: {exc}") from exc


class ExternalCarrierRequest(BaseModel):
    synthesis: SynthesisRequest
    source_provider: str = Field(min_length=1)
    sample_rate: int = Field(gt=0)
    pcm_s16le_base64: str = Field(min_length=4)


@router.post("/finalize")
def finalize_external(request: ExternalCarrierRequest) -> Response:
    try:
        raw = base64.b64decode(request.pcm_s16le_base64, validate=True)
        waveform = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        generated = get_engine().finalize_external(
            request.synthesis,
            waveform,
            request.sample_rate,
            request.source_provider,
        )
        return _audio_response(generated, request.synthesis.voice_id)
    except (FileNotFoundError, KeyError, ValueError, DigitalTwinInvariantError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Twin conversion unavailable: {exc}") from exc
