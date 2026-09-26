from __future__ import annotations

from jiwer import cer as _cer


def character_error_rate(reference: str, hypothesis: str) -> float:
    return float(_cer(reference, hypothesis))
