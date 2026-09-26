from __future__ import annotations

import csv
from pathlib import Path


def create_prosody_abx_trials(
    matched: list[tuple[str, str, str]],
    output_csv: str | Path,
) -> Path:
    path = Path(output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["trial_id", "A_target_delivery", "B_other_delivery", "X_clone", "correct"])
        for i, (a, b, x) in enumerate(matched):
            writer.writerow([i, a, b, x, "A"])
    return path
