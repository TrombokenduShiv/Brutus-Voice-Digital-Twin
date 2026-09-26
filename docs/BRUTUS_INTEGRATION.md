# BRUTUS Integration

BRUTUS must integrate only with the Voice Digital Twin service, never directly with Gemini, Qwen, or another TTS provider.

Production path:

    BRUTUS reasoning text
      -> POST /v1/synthesize or WS /v1/tts/stream
      -> VoiceTwinEngine
      -> Gemini by default
      -> DigitalTwinFinalizer
      -> Device Renderer
      -> 24 kHz mono PCM16
      -> existing BRUTUS robot audio pacer

The response includes digital_twin=true metadata. A request fails instead of returning raw provider voice when no target-voice replication or post-conversion path is available.

The request supports provider override for experiments, but voice_id always refers to the local enrolled VoiceDNA profile. Provider-specific IDs are internal bindings stored inside that profile.
