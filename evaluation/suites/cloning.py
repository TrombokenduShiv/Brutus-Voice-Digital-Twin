from __future__ import annotations

from pathlib import Path

import numpy as np

from evaluation.metrics.cer import character_error_rate
from evaluation.metrics.f0 import f0_metrics
from evaluation.metrics.pauses import pause_metrics
from evaluation.metrics.speaker_ecapa import cosine_similarity, identity_retention
from evaluation.metrics.speaker_ssl import (
    cosine_similarity as ssl_cosine,
    ssl_identity_retention,
)
from evaluation.metrics.spectral import mel_spectral_distance
from evaluation.metrics.wer import word_error_rate
from voice_twin.audio.io import load_audio
from voice_twin.audio.resample import resample
from voice_twin.identity.ecapa_encoder import EcapaSpeakerEncoder
from voice_twin.identity.ssl_encoder import SSLSpeakerEncoder
from voice_twin.prosody.f0 import extract_f0
from voice_twin.prosody.pause_detector import detect_pauses


def evaluate_clone(
    reference_path: str | Path,
    generated_path: str | Path,
    *,
    reference_text: str | None = None,
    hypothesis_text: str | None = None,
    self_reference_path: str | Path | None = None,
    heavy_identity: bool = False,
    device: str = "cpu",
) -> dict[str, float]:
    ref, ref_sr = load_audio(reference_path)
    gen, gen_sr = load_audio(generated_path)
    target_sr = 24000
    ref = resample(ref, ref_sr, target_sr)
    gen = resample(gen, gen_sr, target_sr)

    metrics = {}
    metrics.update(
        f0_metrics(
            extract_f0(ref, target_sr),
            extract_f0(gen, target_sr),
        )
    )
    metrics.update(
        pause_metrics(
            detect_pauses(ref, target_sr),
            detect_pauses(gen, target_sr),
        )
    )
    metrics.update(mel_spectral_distance(ref, gen, target_sr))

    if reference_text is not None and hypothesis_text is not None:
        metrics["wer_clean"] = word_error_rate(reference_text, hypothesis_text)
        metrics["cer_clean"] = character_error_rate(reference_text, hypothesis_text)

    if heavy_identity:
        ecapa = EcapaSpeakerEncoder(device=device)
        e_ref = ecapa.encode(ref, target_sr)
        e_gen = ecapa.encode(gen, target_sr)
        metrics["ecapa_cosine"] = cosine_similarity(e_ref, e_gen)

        ssl = SSLSpeakerEncoder(device=device)
        s_ref = ssl.encode(ref, target_sr)
        s_gen = ssl.encode(gen, target_sr)
        metrics["ssl_cosine"] = ssl_cosine(s_ref, s_gen)

        if self_reference_path:
            self_audio, self_sr = load_audio(self_reference_path)
            self_audio = resample(self_audio, self_sr, target_sr)
            e_self = ecapa.encode(self_audio, target_sr)
            s_self = ssl.encode(self_audio, target_sr)
            metrics["ecapa_identity_retention"] = identity_retention(
                metrics["ecapa_cosine"],
                cosine_similarity(e_ref, e_self),
            )
            metrics["ssl_identity_retention"] = ssl_identity_retention(
                metrics["ssl_cosine"],
                ssl_cosine(s_ref, s_self),
            )
    return metrics
