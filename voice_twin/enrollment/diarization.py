from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from voice_twin.identity.ecapa_encoder import EcapaSpeakerEncoder


@dataclass(slots=True)
class DiarizationResult:
    labels: list[int]
    starts_s: list[float]
    ends_s: list[float]
    dominant_ratio: float
    speaker_count: int

    @property
    def enrollment_clean(self) -> bool:
        return self.speaker_count == 1 or self.dominant_ratio >= 0.90


class SpeakerConsistencyDiarizer:
    """Lightweight enrollment contamination detector based on sliding ECAPA embeddings."""

    def __init__(
        self,
        encoder: EcapaSpeakerEncoder | None = None,
        window_s: float = 1.5,
        hop_s: float = 0.75,
        distance_threshold: float = 0.35,
    ):
        self.encoder = encoder or EcapaSpeakerEncoder()
        self.window_s = window_s
        self.hop_s = hop_s
        self.distance_threshold = distance_threshold

    def analyze(self, audio: np.ndarray, sample_rate: int) -> DiarizationResult:
        from sklearn.cluster import AgglomerativeClustering

        win = max(1, int(self.window_s * sample_rate))
        hop = max(1, int(self.hop_s * sample_rate))
        starts = list(range(0, max(1, len(audio) - win + 1), hop))
        if not starts:
            starts = [0]
        embeddings, start_s, end_s = [], [], []
        for start in starts:
            chunk = audio[start : min(len(audio), start + win)]
            if len(chunk) < sample_rate // 2:
                continue
            emb = self.encoder.encode(chunk)
            emb = emb / max(float(np.linalg.norm(emb)), 1e-8)
            embeddings.append(emb)
            start_s.append(start / sample_rate)
            end_s.append(min(len(audio), start + win) / sample_rate)
        if len(embeddings) <= 1:
            return DiarizationResult([0] * len(embeddings), start_s, end_s, 1.0, 1)
        x = np.stack(embeddings)
        clustering = AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage="average",
            distance_threshold=self.distance_threshold,
        )
        labels = clustering.fit_predict(x).tolist()
        counts = np.bincount(labels)
        return DiarizationResult(
            labels=labels,
            starts_s=start_s,
            ends_s=end_s,
            dominant_ratio=float(counts.max() / counts.sum()),
            speaker_count=int(len(counts)),
        )
