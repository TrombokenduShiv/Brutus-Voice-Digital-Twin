from __future__ import annotations

from datetime import datetime, timezone

from voice_twin.profiles.provenance import sha256_file
from voice_twin.schemas import ConsentRecord


DEFAULT_STATEMENT = "I consent to creation and use of this enrolled synthetic voice profile."


def create_consent_record(speaker_id: str, source_audio: str, verified: bool, statement: str = DEFAULT_STATEMENT) -> ConsentRecord:
    return ConsentRecord(
        speaker_id=speaker_id,
        statement=statement,
        captured_at=datetime.now(timezone.utc).isoformat(),
        source_sha256=sha256_file(source_audio),
        verified=verified,
    )


def require_verified(record: ConsentRecord) -> None:
    if not record.verified:
        raise PermissionError("voice enrollment requires explicit verified consent")
