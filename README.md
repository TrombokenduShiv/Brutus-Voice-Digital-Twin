# BRUTUS Voice Digital Twin

Standalone HDVR voice-digital-twin engine for BRUTUS.

## Non-negotiable runtime invariant

Every externally emitted waveform must be Digital Twin output.

The TTS model is a provider, not the owner of speaker identity:

    text
      -> HDVR / VoiceDNA plan
      -> selected TTS provider
      -> DigitalTwinFinalizer
      -> device renderer
      -> 24 kHz PCM16
      -> BRUTUS

Gemini 3.8 Flash TTS is the default provider. Qwen3-TTS remains a local/research provider. Additional TTS engines can implement the same provider interface.

A provider can satisfy the Digital Twin invariant in one of two ways:

1. Native replicated-voice mode: the provider synthesizes with a consent-backed target-voice binding. Gemini Voice Replication and Qwen reference cloning use this path.
2. Post-conversion mode: an arbitrary TTS provider produces carrier speech and a configured neural TwinConversionBackend converts that waveform into the enrolled VoiceDNA identity.

If neither condition is true, synthesis fails closed. Non-twin provider audio is never returned by the public API.

## Gemini setup

Install the project and set credentials:

    pip install -e ".[dev]"
    export GEMINI_API_KEY=...
    export BVT_DEFAULT_TTS_PROVIDER=gemini

Create the local VoiceDNA profile first, then create a consent-backed Gemini replicated voice and bind it:

    python scripts/register_gemini_voice.py       --voice-id target_speaker       --source-audio reference_speaker.wav       --consent-audio speaker_consent.wav       --display-name "BRUTUS Digital Twin"

Synthesize:

    python scripts/synthesize.py       --voice target_speaker       --text "BRUTUS is online."

## Provider-independent conversion

For TTS providers without native target-voice replication, configure a trained conversion service:

    export BVT_TWIN_CONVERTER_URL=http://127.0.0.1:8790/v1/convert

The service contract receives source PCM, VoiceDNA and the HDVR plan and must return target Digital Twin PCM.

## Architecture

VoiceDNA separates stable identity from dynamic behaviour:

- IdentityCore and VocalProfile
- multi-reference SpeakerMemory
- phoneme/context AccentAtlas
- ProsodyMemory and time-varying trajectory
- breath/non-verbal EventSignature
- provider-specific replicated-voice bindings
- HDVR style/conditioning plan

See docs/ARCHITECTURE.md, docs/TRAINING.md, docs/EVALUATION.md, and docs/BRUTUS_INTEGRATION.md.
