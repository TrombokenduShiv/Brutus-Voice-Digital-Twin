from __future__ import annotations

import numpy as np

from voice_twin.identity.identity_core import aggregate_identity
from voice_twin.identity.vocal_profile import vocal_profile
from voice_twin.profiles.voice_dna import VoiceDNA


def build_voice_dna(
    speaker_id: str,
    embeddings: np.ndarray,
    f0_hz: np.ndarray,
    energy: np.ndarray,
    accent_atlas: dict | None = None,
    prosody_memory: list[dict] | None = None,
    event_signature: dict | None = None,
) -> VoiceDNA:
    embeddings = np.asarray(embeddings, dtype=np.float32)
    f0_hz = np.asarray(f0_hz, dtype=np.float32)
    energy = np.asarray(energy, dtype=np.float32)
    identity = aggregate_identity(embeddings)
    vocal = vocal_profile(f0_hz, energy)
    voiced = f0_hz[f0_hz > 0]
    style_profile = {
        "f0_mean_hz": float(voiced.mean()) if len(voiced) else 0.0,
        "f0_std_hz": float(voiced.std()) if len(voiced) else 0.0,
        "energy_mean": float(energy.mean()) if len(energy) else 0.0,
        "voicing_ratio": float(len(voiced) / max(1, len(f0_hz))),
        "pace_ratio": 1.0,
        "breath_probability": float(
            (event_signature or {}).get("breath_probability", 0.0)
        ),
    }
    return VoiceDNA(
        speaker_id=speaker_id,
        identity_core=identity,
        vocal_profile=vocal,
        identity_memory=embeddings,
        accent_atlas=accent_atlas or {},
        prosody_memory=prosody_memory or [],
        event_signature=event_signature or {},
        style_profile=style_profile,
    )
