from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["enrollment"])


class EnrollmentRequest(BaseModel):
    speaker_id: str
    source_audio: str
    consent_verified: bool


@router.post("/voices/enroll")
def enroll(req: EnrollmentRequest) -> dict:
    if not req.consent_verified:
        return {"accepted": False, "reason": "verified consent is required"}
    return {"accepted": True, "speaker_id": req.speaker_id, "next": "run enrollment pipeline"}
