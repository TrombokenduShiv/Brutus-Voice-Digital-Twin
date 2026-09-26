from __future__ import annotations


def prosody_metadata(sequence: int, f0_hz: float, energy: float, breath_probability: float, emotion: str | None = None) -> dict:
    return {
        "type": "prosody",
        "sequence": sequence,
        "f0": f0_hz,
        "energy": energy,
        "breath_probability": breath_probability,
        "emotion": emotion,
        "mouth_open": max(0.0, min(1.0, energy * 4.0)),
    }
