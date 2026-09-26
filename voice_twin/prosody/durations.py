from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class TimedToken:
    token: str
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)


def duration_mae_ms(reference: list[TimedToken], predicted: list[TimedToken]) -> float:
    pairs = min(len(reference), len(predicted))
    if not pairs:
        return 0.0
    return sum(abs(reference[i].duration_s - predicted[i].duration_s) for i in range(pairs)) * 1000.0 / pairs
