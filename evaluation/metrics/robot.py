from __future__ import annotations

from evaluation.metrics.spectral import mel_spectral_distance


def over_air_metrics(
    reference_audio,
    robot_recording,
    sample_rate: int,
    identity_retention: float | None = None,
) -> dict[str, float]:
    metrics = {
        f"robot_{k}": v
        for k, v in mel_spectral_distance(
            reference_audio, robot_recording, sample_rate
        ).items()
    }
    if identity_retention is not None:
        metrics["over_air_identity_retention"] = float(identity_retention)
    return metrics
