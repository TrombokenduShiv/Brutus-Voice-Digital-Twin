from __future__ import annotations

import numpy as np

from voice_twin.audio.normalize import to_pcm16
from voice_twin.schemas import AudioFrame
from voice_twin.streaming.chunker import chunk_audio


def audio_frames(audio: np.ndarray, sample_rate: int = 24000, chunk_ms: int = 60):
    for sequence, chunk in enumerate(chunk_audio(audio, sample_rate, chunk_ms)):
        yield AudioFrame(
            sequence=sequence,
            sample_rate=sample_rate,
            duration_ms=len(chunk) * 1000.0 / sample_rate,
            pcm=to_pcm16(chunk),
        )
