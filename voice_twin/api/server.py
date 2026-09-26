from __future__ import annotations

from fastapi import FastAPI

from voice_twin.api.routes_enrollment import router as enrollment_router
from voice_twin.api.routes_eval import router as eval_router
from voice_twin.api.routes_voice import router as voice_router
from voice_twin.api.websocket import router as websocket_router

app = FastAPI(title="BRUTUS Voice Digital Twin", version="0.1.0")
app.include_router(enrollment_router, prefix="/v1")
app.include_router(voice_router, prefix="/v1")
app.include_router(eval_router, prefix="/v1")
app.include_router(websocket_router, prefix="/v1")


@app.get("/health")
def health() -> dict:
    return {"ok": True, "service": "brutus-voice-twin"}
