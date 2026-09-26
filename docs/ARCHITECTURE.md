# Architecture

BRUTUS Voice Digital Twin implements a Hierarchical Dynamic Voice Representation (HDVR).

The representation is factored into stable identity, vocal profile, multi-reference speaker memory, a phoneme-conditioned Accent Atlas, a time-varying prosody trajectory, explicit breath/non-verbal events, and retrieved speaker-style memories.

Two generation routes are intentionally separate:

1. Digital-twin TTS predicts how an enrolled speaker would likely deliver unseen text.
2. Performance replication uses a supplied reference performance and preserves measured timing/prosody as explicit controls.

The first acoustic baseline is Qwen3-TTS Base. HDVR remains model-agnostic: the acoustic backend consumes conditioning and returns waveform/audio tokens. The robot boundary remains 24 kHz mono PCM16.

No code path treats a single fixed speaker embedding as the complete speaker representation.
