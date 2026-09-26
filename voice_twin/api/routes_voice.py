from __future__ import annotations

from fastapi import APIRouter, HTTPException, Response

from voice_twin.api.runtime import get_engine
from voice_twin.audio.normalize import to_pcm16
from voice_twin.conversion.twin_converter import DigitalTwinInvariantError
from voice_twin.schemas import SynthesisRequest

router = APIRouter(tags=["voice"])


@router.post("/synthesize")
def synthesize(request: SynthesisRequest) -> Response:
    try:
        generated = get_engine().synthesize(request)
    except (FileNotFoundError, KeyError, ValueError, DigitalTwinInvariantError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"TTS runtime unavailable: {exc}") from exc

    return Response(
        content=to_pcm16(generated.waveform),
        media_type="audio/l16",
        headers={
            "X-BVT-Digital-Twin": "true",
            "X-BVT-Provider": generated.provider,
            "X-BVT-Voice-Id": request.voice_id,
            "X-BVT-Sample-Rate": str(generated.sample_rate),
        },
    )
