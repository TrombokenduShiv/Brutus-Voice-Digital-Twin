from __future__ import annotations

from jiwer import wer as _wer


def word_error_rate(reference: str, hypothesis: str) -> float:
    return float(_wer(reference, hypothesis))
