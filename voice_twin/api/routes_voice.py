from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(tags=["voice"])


class SynthesisPreview(BaseModel):
    text: str
    voice_id: str
    language: str = "English"
    mode: str = "twin"


@router.post("/synthesize")
def synthesize_preview(request: SynthesisPreview) -> dict:
    # Heavy model construction is deliberately injected by deployment code, not import time.
    raise HTTPException(
        status_code=503,
        detail="No acoustic backend is loaded. Start the runtime with a configured model backend.",
    )
