from __future__ import annotations

import hmac
import os


def verify_api_token(candidate: str | None) -> bool:
    expected = os.getenv("BVT_API_TOKEN")
    if not expected:
        return True
    return bool(candidate) and hmac.compare_digest(candidate, expected)
