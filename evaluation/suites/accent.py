from __future__ import annotations

from evaluation.metrics.accent import (
    accent_feature_distance,
    whole_utterance_accent_features,
)
from voice_twin.audio.io import load_audio
from voice_twin.audio.resample import resample


def evaluate_accent(reference_path, generated_path) -> dict[str, float]:
    r, rs = load_audio(reference_path)
    g, gs = load_audio(generated_path)
    sr = 24000
    r, g = resample(r, rs, sr), resample(g, gs, sr)
    return accent_feature_distance(
        whole_utterance_accent_features(r, sr),
        whole_utterance_accent_features(g, sr),
    )
