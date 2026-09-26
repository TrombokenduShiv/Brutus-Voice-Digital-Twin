# BRUTUS Voice Digital Twin

Standalone high-fidelity voice cloning and speaker-behaviour modeling service designed to integrate with BRUTUS through a streaming TTS provider.

## Architecture

The system separates stable identity from time-varying behaviour:

- IdentityCore and VocalProfile model stable speaker acoustics.
- SpeakerMemory keeps multiple reference states instead of averaging one speaker vector.
- AccentAtlas models phoneme/allophone-conditioned pronunciation.
- ProsodyMemory and ProsodyTrajectory model F0, energy, duration, pauses and voicing through time.
- EventSignature models breath and non-verbal events.
- HDVR fuses those controls before the acoustic backend.
- Qwen3-TTS is the first production baseline; F5/OpenVoice remain benchmark adapters.
- Performance mode preserves reference timing/prosody; twin mode predicts speaker behaviour for unseen text.
- The public streaming contract is 24 kHz mono PCM16 plus timing/prosody metadata for BRUTUS.

## Development state

This repository contains an executable research architecture and evaluation harness. Foundation-model weights and trained speaker adapters are not stored in Git. Training data must be consented and supplied separately.

## Quick start

    python -m venv .venv
    .venv/bin/pip install -e ".[dev]"
    pytest
    uvicorn voice_twin.api.server:app --reload

Windows PowerShell activation:

    .venv\Scripts\Activate.ps1

## Main commands

    python scripts/prepare_dataset.py --help
    python scripts/enroll_voice.py --help
    python scripts/train.py --help
    python scripts/synthesize.py --help
    python scripts/evaluate.py --help

See docs/ARCHITECTURE.md, docs/TRAINING.md and docs/EVALUATION.md.
