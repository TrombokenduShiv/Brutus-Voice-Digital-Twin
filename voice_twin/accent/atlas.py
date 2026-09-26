from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class AccentCell:
    count: int = 0
    duration_ms_mean: float = 0.0
    f0_delta_mean: float = 0.0
    energy_delta_mean: float = 0.0
    features: dict[str, float] = field(default_factory=dict)


class AccentAtlas:
    def __init__(self):
        self.cells: dict[str, AccentCell] = {}

    @staticmethod
    def key(phoneme: str, context: str) -> str:
        return f"{phoneme}|{context}"

    def update(self, phoneme: str, context: str, duration_ms: float, f0_delta: float = 0.0, energy_delta: float = 0.0) -> None:
        key = self.key(phoneme, context)
        cell = self.cells.setdefault(key, AccentCell())
        n = cell.count
        cell.duration_ms_mean = (cell.duration_ms_mean * n + duration_ms) / (n + 1)
        cell.f0_delta_mean = (cell.f0_delta_mean * n + f0_delta) / (n + 1)
        cell.energy_delta_mean = (cell.energy_delta_mean * n + energy_delta) / (n + 1)
        cell.count += 1

    def get(self, phoneme: str, context: str) -> AccentCell | None:
        return self.cells.get(self.key(phoneme, context))

    def to_dict(self) -> dict[str, Any]:
        return {k: asdict(v) for k, v in self.cells.items()}
