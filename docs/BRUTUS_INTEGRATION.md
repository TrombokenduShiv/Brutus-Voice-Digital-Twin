# BRUTUS Integration

BRUTUS should depend only on a TTS provider boundary.

Input: text, voice_id, language, optional emotion/style and abort/cancellation state.

Output: streamed 24 kHz mono PCM16 frames plus optional prosody metadata.

Do not move HDVR/model internals into Electron. Run this project as a local/Brain-Node service and connect it to the existing BRUTUS robot PCM pacing path.
