from __future__ import annotations

import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from voice_twin.api.runtime import get_engine
from voice_twin.schemas import SynthesisRequest
from voice_twin.streaming.pcm_stream import audio_frames

router = APIRouter(tags=["streaming"])


@router.websocket("/tts/stream")
async def tts_stream(ws: WebSocket):
    await ws.accept()
    try:
        payload = await ws.receive_json()
        request = SynthesisRequest.model_validate(payload)
        generated = await asyncio.to_thread(get_engine().synthesize, request)
        await ws.send_json({
            "type": "ready",
            "format": "pcm_s16le",
            "sample_rate": generated.sample_rate,
            "channels": 1,
            "provider": generated.provider,
            "digital_twin": generated.digital_twin,
            "voice_id": request.voice_id,
        })
        for frame in audio_frames(generated.waveform, generated.sample_rate):
            await ws.send_bytes(frame.pcm)
        await ws.send_json({"type": "complete", "digital_twin": True})
    except WebSocketDisconnect:
        return
    except Exception as exc:
        await ws.send_json({"type": "error", "message": str(exc)})
    finally:
        await ws.close()
