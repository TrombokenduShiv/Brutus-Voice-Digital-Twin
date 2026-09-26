from __future__ import annotations

import csv
import random
from pathlib import Path


def create_abx_trials(
    target_files: list[str],
    impostor_files: list[str],
    clone_files: list[str],
    output_csv: str | Path,
    seed: int = 42,
) -> Path:
    if not (target_files and impostor_files and clone_files):
        raise ValueError("ABX requires target, impostor, and clone files")
    rng = random.Random(seed)
    rows = []
    for i, clone in enumerate(clone_files):
        rows.append(
            [
                i,
                rng.choice(target_files),
                rng.choice(impostor_files),
                clone,
                "A",
            ]
        )
    rng.shuffle(rows)
    path = Path(output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["trial_id", "A_target", "B_impostor", "X_clone", "correct"])
        writer.writerows(rows)
    return path


def score_abx(correct: int, trials: int) -> float:
    return correct / trials if trials else 0.0
