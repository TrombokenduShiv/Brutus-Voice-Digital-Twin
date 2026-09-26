from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(tags=["evaluation"])


@router.get("/evaluation/metrics")
def metrics() -> dict:
    return {
        "identity": ["ecapa", "ssl", "identity_retention"],
        "content": ["wer", "cer"],
        "prosody": ["f0_correlation", "duration_mae", "pause_metrics"],
        "runtime": ["ttfa_ms", "realtime_factor"],
    }
