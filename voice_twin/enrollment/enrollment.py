from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from voice_twin.audio.io import load_audio
from voice_twin.audio.normalize import peak_normalize
from voice_twin.audio.quality import AudioQuality, analyze_quality
from voice_twin.audio.resample import resample
from voice_twin.constants import MASTER_SAMPLE_RATE
from voice_twin.enrollment.consent import create_consent_record, require_verified
from voice_twin.prosody.energy import extract_energy
from voice_twin.prosody.f0 import extract_f0
from voice_twin.schemas import ConsentRecord


@dataclass(slots=True)
class EnrollmentSample:
    path: str
    audio: np.ndarray
    sample_rate: int
    quality: AudioQuality


class EnrollmentPipeline:
    def prepare(self, path: str | Path) -> EnrollmentSample:
        audio, sr = load_audio(path)
        audio = resample(audio, sr, MASTER_SAMPLE_RATE)
        audio = peak_normalize(audio)
        q = analyze_quality(audio, MASTER_SAMPLE_RATE)
        return EnrollmentSample(str(path), audio, MASTER_SAMPLE_RATE, q)

    def consent(self, speaker_id: str, source_audio: str | Path, verified: bool) -> ConsentRecord:
        record = create_consent_record(speaker_id, str(source_audio), verified)
        require_verified(record)
        return record

    def acoustic_tracks(self, sample: EnrollmentSample) -> tuple[np.ndarray, np.ndarray]:
        return extract_f0(sample.audio, sample.sample_rate), extract_energy(sample.audio, sample.sample_rate)
