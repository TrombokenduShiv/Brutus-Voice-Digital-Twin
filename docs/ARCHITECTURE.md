# Architecture

BRUTUS Voice Digital Twin implements a Hierarchical Dynamic Voice Representation (HDVR).

The representation is factored into stable identity, vocal profile, multi-reference speaker memory, a phoneme-conditioned Accent Atlas, a time-varying prosody trajectory, explicit breath/non-verbal events, retrieved speaker-style memories, and provider bindings.

## Mandatory output path

No TTS provider is an external output boundary.

    Request
      -> VoiceDNA profile load
      -> HDVR DigitalTwinPlan
      -> TTS provider
      -> DigitalTwinFinalizer
           -> native replicated voice accepted
           OR
           -> trained TwinConversionBackend
           OR
           -> FAIL CLOSED
      -> robot/device acoustic renderer
      -> PCM stream

This means Gemini, Qwen, F5, system TTS, or any future model can provide the acoustic carrier, but provider audio is never allowed to bypass DigitalTwinFinalizer.

## Gemini-first production path

Gemini is the default TTS provider. An enrolled VoiceDNA profile stores a provider binding containing the consent-backed Gemini replicated voice ID/key. HDVR produces the style/prosody instruction; Gemini produces target-identity speech; DigitalTwinFinalizer verifies the invariant before output.

## Arbitrary providers

Providers that cannot natively synthesize the enrolled speaker are permitted only when a trained TwinConversionBackend is configured. The included HTTP adapter defines that model boundary. Without it, carrier speech is rejected.

## Research routes

Qwen3-TTS remains a native reference-cloning baseline. Performance-replication research remains separate from novel-text synthesis, but all production emission still uses the same finalizer.

The physical BRUTUS boundary remains 24 kHz mono PCM16.
