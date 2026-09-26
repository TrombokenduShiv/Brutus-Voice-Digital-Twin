from __future__ import annotations


def evaluate_streaming(
    *,
    request_s: float,
    first_pcm_s: float,
    completion_s: float,
    audio_duration_s: float,
    underruns: int = 0,
) -> dict[str, float]:
    compute = max(0.0, completion_s - request_s)
    return {
        "warm_ttfa_ms": max(0.0, first_pcm_s - request_s) * 1000.0,
        "realtime_factor": compute / max(audio_duration_s, 1e-8),
        "underruns_per_minute": underruns / max(audio_duration_s / 60.0, 1e-8),
    }
