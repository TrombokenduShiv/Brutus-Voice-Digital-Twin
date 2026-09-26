from __future__ import annotations

from fastapi import APIRouter, WebSocket

router = APIRouter(tags=["streaming"])


@router.websocket("/tts/stream")
async def tts_stream(ws: WebSocket):
    await ws.accept()
    await ws.send_json({
        "type": "ready",
        "format": "pcm_s16le",
        "sample_rate": 24000,
        "channels": 1,
        "message": "Runtime backend must be injected by deployment entrypoint.",
    })
    await ws.close()
