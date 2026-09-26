from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class ConversionDataset(Dataset):
    def __init__(self, manifest: str | Path):
        self.manifest = Path(manifest)
        self.rows = [
            json.loads(line)
            for line in self.manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = Path(row["feature_path"])
        if not path.is_absolute():
            path = (self.manifest.parent / path).resolve()
        with np.load(path, allow_pickle=False) as data:
            return {
                "source_mel": torch.from_numpy(data["source_mel"]).float(),
                "target_mel": torch.from_numpy(data["target_mel"]).float(),
                "identity": torch.from_numpy(data["identity"]).float(),
                "vocal": torch.from_numpy(data["vocal"]).float(),
                "style": torch.from_numpy(data["style"]).float(),
                "duration_s": float(row.get("duration", 0.0)),
                "id": str(row.get("id", index)),
            }


def collate_conversion(batch: list[dict]) -> dict:
    lengths = torch.tensor([len(x["source_mel"]) for x in batch], dtype=torch.long)
    max_len = int(lengths.max())
    mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)
    return {
        "source_mel": pad_sequence(
            [x["source_mel"] for x in batch], batch_first=True
        ),
        "target_mel": pad_sequence(
            [x["target_mel"] for x in batch], batch_first=True
        ),
        "identity": torch.stack([x["identity"] for x in batch]),
        "vocal": torch.stack([x["vocal"] for x in batch]),
        "style": torch.stack([x["style"] for x in batch]),
        "frame_mask": mask,
        "lengths": lengths,
        "id": [x["id"] for x in batch],
    }
