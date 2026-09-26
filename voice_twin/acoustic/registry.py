from __future__ import annotations

from voice_twin.acoustic.base import AcousticBackend


class ProviderRegistry:
    def __init__(self):
        self._providers: dict[str, AcousticBackend] = {}

    def register(self, provider: AcousticBackend) -> None:
        if not provider.provider_name:
            raise ValueError("provider_name is required")
        self._providers[provider.provider_name] = provider

    def get(self, name: str) -> AcousticBackend:
        try:
            return self._providers[name]
        except KeyError as exc:
            raise KeyError(f"unknown TTS provider: {name}") from exc

    def names(self) -> list[str]:
        return sorted(self._providers)
