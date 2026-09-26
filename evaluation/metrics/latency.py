from __future__ import annotations


def realtime_factor(compute_seconds: float, audio_seconds: float) -> float:
    return compute_seconds / audio_seconds if audio_seconds > 0 else float("inf")
