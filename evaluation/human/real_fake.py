from __future__ import annotations

import csv
import random
from pathlib import Path

from evaluation.human.statistics import real_fake_equivalent, wilson_interval


def create_real_fake_trials(
    real_files: list[str],
    synthetic_files: list[str],
    output_csv: str | Path,
    seed: int = 42,
) -> Path:
    rows = [(path, "real") for path in real_files] + [
        (path, "synthetic") for path in synthetic_files
    ]
    random.Random(seed).shuffle(rows)
    path = Path(output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["trial_id", "audio", "ground_truth"])
        for i, (audio, label) in enumerate(rows):
            writer.writerow([i, audio, label])
    return path


def score_real_fake(correct: int, trials: int, margin: float = 0.05) -> dict:
    lo, hi = wilson_interval(correct, trials)
    return {
        "real_fake_accuracy": correct / trials,
        "real_fake_ci_low": lo,
        "real_fake_ci_high": hi,
        "real_fake_equivalent": real_fake_equivalent(correct, trials, margin),
    }
