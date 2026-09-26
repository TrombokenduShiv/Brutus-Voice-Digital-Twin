from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class VoiceBank:
    speaker_id: str
    events: dict[str, list[str]] = field(default_factory=dict)

    def add(self, kind: str, audio_path: str | Path) -> None:
        self.events.setdefault(kind, []).append(str(audio_path))

    def samples(self, kind: str) -> list[str]:
        return list(self.events.get(kind, []))
