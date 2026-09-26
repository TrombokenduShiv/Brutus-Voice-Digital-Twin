from __future__ import annotations

from collections import defaultdict

import numpy as np

from voice_twin.accent.acoustic_features import FEATURE_NAMES, extract_accent_features
from voice_twin.enrollment.alignment import TokenAlignment


class AccentAtlasBuilder:
    def __init__(self):
        self._values: dict[str, list[np.ndarray]] = defaultdict(list)

    @staticmethod
    def context(alignment: list[TokenAlignment], index: int) -> str:
        left = alignment[index - 1].token if index > 0 else "<BOS>"
        right = alignment[index + 1].token if index + 1 < len(alignment) else "<EOS>"
        return f"{left}_{right}"

    def add(
        self,
        audio: np.ndarray,
        sample_rate: int,
        alignment: list[TokenAlignment],
    ) -> None:
        for i, item in enumerate(alignment):
            start = max(0, int(item.start_s * sample_rate))
            end = min(len(audio), max(start + 1, int(item.end_s * sample_rate)))
            features = extract_accent_features(audio[start:end], sample_rate)
            self._values[f"{item.token}|{self.context(alignment, i)}"].append(features)

    def build(self) -> dict:
        cells = {}
        for key, values in self._values.items():
            x = np.stack(values)
            cells[key] = {
                "count": int(len(x)),
                "mean": x.mean(axis=0).astype(float).tolist(),
                "std": x.std(axis=0).astype(float).tolist(),
                "feature_names": list(FEATURE_NAMES),
            }
        return cells
