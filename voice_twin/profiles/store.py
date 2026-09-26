from __future__ import annotations

import json
from pathlib import Path

from cryptography.fernet import Fernet

from voice_twin.profiles.voice_dna import VoiceDNA


class VoiceProfileStore:
    def __init__(self, root: str | Path, encryption_key: bytes | None = None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._fernet = Fernet(encryption_key) if encryption_key else None

    def _path(self, voice_id: str) -> Path:
        safe = "".join(c for c in voice_id if c.isalnum() or c in "-_")
        if not safe:
            raise ValueError("invalid voice id")
        return self.root / f"{safe}.json.enc"

    def save(self, profile: VoiceDNA) -> Path:
        raw = json.dumps(profile.to_jsonable(), separators=(",", ":")).encode()
        if self._fernet:
            raw = self._fernet.encrypt(raw)
        path = self._path(profile.speaker_id)
        path.write_bytes(raw)
        return path

    def load(self, voice_id: str) -> VoiceDNA:
        raw = self._path(voice_id).read_bytes()
        if self._fernet:
            raw = self._fernet.decrypt(raw)
        return VoiceDNA.from_jsonable(json.loads(raw))

    def delete(self, voice_id: str) -> None:
        self._path(voice_id).unlink(missing_ok=True)

    def list_ids(self) -> list[str]:
        return sorted(p.name.removesuffix(".json.enc") for p in self.root.glob("*.json.enc"))
