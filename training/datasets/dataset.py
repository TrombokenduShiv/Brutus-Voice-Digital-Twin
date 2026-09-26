from __future__ import annotations

import json
from pathlib import Path

from torch.utils.data import Dataset


class ManifestDataset(Dataset):
    def __init__(self, manifest: str | Path):
        with Path(manifest).open("r", encoding="utf-8") as f:
            self.rows = [json.loads(line) for line in f if line.strip()]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return self.rows[index]
