from __future__ import annotations

import base64
import os
from pathlib import Path


class GeminiVoiceManager:
    def __init__(self, api_key: str | None = None, model_id: str = "gemini-3.8-flash-tts"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_id = model_id
        self._client = None

    def _load(self):
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=self.api_key) if self.api_key else genai.Client()
        return self._client

    @staticmethod
    def _audio(path: str | Path) -> dict:
        p = Path(path)
        suffix = p.suffix.lower()
        mime = "audio/wav" if suffix == ".wav" else "audio/mpeg" if suffix in {".mp3", ".mpeg"} else "audio/wav"
        return {
            "mime_type": mime,
            "data": base64.b64encode(p.read_bytes()).decode("utf-8"),
        }

    def create_replicated_voice(
        self,
        source_audio: str | Path,
        consent_audio: str | Path,
        display_name: str,
        store: bool = True,
    ) -> str:
        voice = self._load().voices.create(
            store=store,
            voice={
                "model": self.model_id,
                "type": "replicated",
                "display_name": display_name,
                "replicated": {
                    "source_audio": self._audio(source_audio),
                    "consent_audio": self._audio(consent_audio),
                },
            },
        )
        identifier = getattr(voice, "id", None) or getattr(voice, "key", None)
        if not identifier:
            raise RuntimeError("Gemini voice replication returned no voice id/key")
        return str(identifier)
