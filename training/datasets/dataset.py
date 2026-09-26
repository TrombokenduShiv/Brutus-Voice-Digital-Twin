from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class ManifestDataset(Dataset):
    def __init__(self, manifest: str | Path):
        self.manifest = Path(manifest)
        with self.manifest.open("r", encoding="utf-8") as f:
            self.rows = [json.loads(line) for line in f if line.strip()]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return self.rows[index]


class FeatureDataset(ManifestDataset):
    """Loads precomputed per-utterance NPZ features referenced by a JSONL manifest."""

    REQUIRED = {
        "token_ids",
        "identity",
        "vocal",
        "speaker_memory",
        "accent_context",
        "prosody_context",
        "prosody_target",
        "event_context",
        "acoustic_target",
        "speaker_target",
        "ssl_target",
        "duration_ms",
        "pause_type",
        "pause_duration_ms",
        "accent_target",
        "event_target",
        "channel_id",
    }

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | float]:
        row = self.rows[index]
        feature_path = Path(row["feature_path"])
        if not feature_path.is_absolute():
            feature_path = (self.manifest.parent / feature_path).resolve()
        with np.load(feature_path, allow_pickle=False) as data:
            missing = self.REQUIRED.difference(data.files)
            if missing:
                raise KeyError(f"{feature_path} missing feature arrays: {sorted(missing)}")
            item = {
                "token_ids": torch.from_numpy(data["token_ids"]).long(),
                "identity": torch.from_numpy(data["identity"]).float(),
                "vocal": torch.from_numpy(data["vocal"]).float(),
                "speaker_memory": torch.from_numpy(data["speaker_memory"]).float(),
                "accent_context": torch.from_numpy(data["accent_context"]).float(),
                "prosody_context": torch.from_numpy(data["prosody_context"]).float(),
                "prosody_target": torch.from_numpy(data["prosody_target"]).float(),
                "event_context": torch.from_numpy(data["event_context"]).float(),
                "acoustic_target": torch.from_numpy(data["acoustic_target"]).float(),
                "speaker_target": torch.from_numpy(data["speaker_target"]).float(),
                "ssl_target": torch.from_numpy(data["ssl_target"]).float(),
                "duration_ms": torch.from_numpy(data["duration_ms"]).float(),
                "pause_type": torch.from_numpy(data["pause_type"]).long(),
                "pause_duration_ms": torch.from_numpy(data["pause_duration_ms"]).float(),
                "accent_target": torch.from_numpy(data["accent_target"]).float(),
                "event_target": torch.from_numpy(data["event_target"]).long(),
                "channel_id": torch.as_tensor(int(data["channel_id"]), dtype=torch.long),
                "duration_s": float(row.get("duration", 0.0)),
                "speaker_id": str(row.get("speaker_id", "")),
                "utterance_id": str(row.get("id", index)),
            }
        return item
