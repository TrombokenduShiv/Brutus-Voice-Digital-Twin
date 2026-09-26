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
    identity = aggregate_identity(embeddings)
    vocal = vocal_profile(f0_hz, energy)
    return VoiceDNA(
        speaker_id=speaker_id,
        identity_core=identity,
        vocal_profile=vocal,
        identity_memory=embeddings,
        accent_atlas=accent_atlas or {},
        prosody_memory=prosody_memory or [],
        event_signature=event_signature or {},
    )
