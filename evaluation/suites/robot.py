from __future__ import annotations

from evaluation.metrics.robot import over_air_metrics
from voice_twin.audio.io import load_audio
from voice_twin.audio.resample import resample


def evaluate_robot(reference_path, robot_recording_path, identity_retention=None):
    r, rs = load_audio(reference_path)
    g, gs = load_audio(robot_recording_path)
    sr = 24000
    return over_air_metrics(
        resample(r, rs, sr),
        resample(g, gs, sr),
        sr,
        identity_retention=identity_retention,
    )
