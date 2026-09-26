from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from voice_twin.constants import VOICE_DNA_VERSION


def _arr(x: np.ndarray | None) -> list[float] | None:
    return None if x is None else np.asarray(x, dtype=np.float32).tolist()


@dataclass(slots=True)
class VoiceDNA:
    speaker_id: str
    identity_core: np.ndarray
    vocal_profile: np.ndarray
    identity_memory: np.ndarray
    accent_atlas: dict[str, Any] = field(default_factory=dict)
    prosody_memory: list[dict[str, Any]] = field(default_factory=list)
    event_signature: dict[str, Any] = field(default_factory=dict)
    pronunciation_dictionary: dict[str, str] = field(default_factory=dict)
    model_version: str = VOICE_DNA_VERSION

    def validate(self) -> None:
        if self.identity_core.ndim != 1:
            raise ValueError("identity_core must be rank-1")
        if self.vocal_profile.ndim != 1:
            raise ValueError("vocal_profile must be rank-1")
        if self.identity_memory.ndim != 2:
            raise ValueError("identity_memory must be [slots, dim]")
        if not np.isfinite(self.identity_core).all():
            raise ValueError("identity_core contains non-finite values")

    def to_jsonable(self) -> dict[str, Any]:
        self.validate()
        return {
            "speaker_id": self.speaker_id,
            "identity_core": _arr(self.identity_core),
            "vocal_profile": _arr(self.vocal_profile),
            "identity_memory": np.asarray(self.identity_memory, dtype=np.float32).tolist(),
            "accent_atlas": self.accent_atlas,
            "prosody_memory": self.prosody_memory,
            "event_signature": self.event_signature,
            "pronunciation_dictionary": self.pronunciation_dictionary,
            "model_version": self.model_version,
        }

    @classmethod
    def from_jsonable(cls, data: dict[str, Any]) -> "VoiceDNA":
        return cls(
            speaker_id=data["speaker_id"],
            identity_core=np.asarray(data["identity_core"], dtype=np.float32),
            vocal_profile=np.asarray(data["vocal_profile"], dtype=np.float32),
            identity_memory=np.asarray(data["identity_memory"], dtype=np.float32),
            accent_atlas=data.get("accent_atlas", {}),
            prosody_memory=data.get("prosody_memory", []),
            event_signature=data.get("event_signature", {}),
            pronunciation_dictionary=data.get("pronunciation_dictionary", {}),
            model_version=data.get("model_version", VOICE_DNA_VERSION),
        )
