# ML Training Runbook

This runbook covers the complete BRUTUS Voice Digital Twin ML pipeline from a fresh machine to trained HDVR and Twin Converter checkpoints.

## 0. Preconditions

Use Python 3.11 or 3.12. A CUDA GPU is strongly recommended for real training.

Required system tools:
- Git
- ffmpeg
- eSpeak or eSpeak-NG
- a working CUDA driver when using NVIDIA GPUs

The production runtime remains Gemini-first. The models trained here provide the Digital Twin identity, prosody, accent, event, and generic-provider conversion layers.

## 1. Environment

Create and activate a virtual environment, then install the ML stack:

    python -m venv .venv

Linux/macOS:

    source .venv/bin/activate

Windows PowerShell:

    .venv\Scripts\Activate.ps1

Install:

    python -m pip install --upgrade pip
    pip install -e ".[dev,ml]"

For the local Twin Converter waveform path also install BigVGAN:

    pip install -e ".[vocoder]"

Optional local Qwen baseline:

    pip install -e ".[qwen]"

Validate:

    python scripts/validate_training_setup.py

Run the deterministic CPU smoke train:

    python scripts/smoke_train.py

Run repository tests:

    pytest

## 2. Prefetch model dependencies

Minimum preprocessing models:

    python scripts/download_models.py --models ctc ssl ecapa

Add the local Twin Converter vocoder:

    python scripts/download_models.py --models ctc ssl ecapa --include-vocoder

Add the optional Qwen baseline:

    python scripts/download_models.py --models ctc ssl ecapa --include-vocoder --include-qwen

## 3. Dataset metadata

Create a CSV or JSONL with at least:

    audio,text,speaker_id,session_id,language

Recommended additional fields:

    id,recording_day,emotion,style,split

Do not split adjacent clips from the same recording session across train and test.

Build leakage-resistant manifests:

    python scripts/build_manifest.py \
      --metadata data/metadata.csv \
      --audio-root data/raw/speakers \
      --output-dir data/manifests

Outputs:

    data/manifests/train.jsonl
    data/manifests/validation.jsonl
    data/manifests/test.jsonl

## 4. Extract HDVR training features

Production-quality preprocessing:

    python scripts/prepare_dataset.py \
      --input-manifest data/manifests/train.jsonl \
      --output-manifest data/processed/train_features.jsonl \
      --feature-dir data/processed/features/train \
      --aligner ctc \
      --ctc-model facebook/wav2vec2-base-960h \
      --device cuda

Repeat for validation and test.

For a pipeline smoke test without downloading ECAPA/WavLM, use:

    python scripts/prepare_dataset.py ... --lightweight --aligner proportional

Do not use lightweight features for final metrics or training claims.

## 5. Train the global HDVR backbone

    python scripts/train.py \
      --stage hdvr \
      --config configs/training/hdvr_pretrain.yaml \
      --model-config configs/models/hdvr.yaml \
      --train-manifest data/processed/train_features.jsonl \
      --validation-manifest data/processed/validation_features.jsonl \
      --output-dir checkpoints/hdvr/global \
      --device cuda

Primary outputs:

    checkpoints/hdvr/global/latest.pt
    checkpoints/hdvr/global/best.pt
    checkpoints/hdvr/global/metrics.jsonl
    checkpoints/hdvr/global/tensorboard/

Monitor:

    tensorboard --logdir checkpoints/hdvr/global/tensorboard

## 6. Stage-specific refinement

Accent:

    python scripts/train.py \
      --stage accent \
      --config configs/training/accent_adapt.yaml \
      --model-config configs/models/hdvr.yaml \
      --train-manifest data/processed/train_features.jsonl \
      --validation-manifest data/processed/validation_features.jsonl \
      --output-dir checkpoints/accent/global \
      --base-checkpoint checkpoints/hdvr/global/best.pt \
      --device cuda

Prosody:

    python scripts/train.py \
      --stage prosody \
      --config configs/training/prosody_adapt.yaml \
      --model-config configs/models/hdvr.yaml \
      --train-manifest data/processed/train_features.jsonl \
      --validation-manifest data/processed/validation_features.jsonl \
      --output-dir checkpoints/prosody/global \
      --base-checkpoint checkpoints/accent/global/best.pt \
      --device cuda

Events:

    python scripts/train.py \
      --stage events \
      --config configs/training/event_model.yaml \
      --model-config configs/models/hdvr.yaml \
      --train-manifest data/processed/train_features.jsonl \
      --validation-manifest data/processed/validation_features.jsonl \
      --output-dir checkpoints/events/global \
      --base-checkpoint checkpoints/prosody/global/best.pt \
      --device cuda

## 7. Speaker adaptation

Create target-speaker-only train and validation feature manifests.

Then:

    python scripts/train.py \
      --stage speaker-adapt \
      --config configs/training/speaker_adapt.yaml \
      --model-config configs/models/hdvr.yaml \
      --train-manifest data/processed/target/train_features.jsonl \
      --validation-manifest data/processed/target/validation_features.jsonl \
      --output-dir checkpoints/adapters/target_speaker \
      --base-checkpoint checkpoints/events/global/best.pt \
      --device cuda

The stage freezes the backbone and trains the lightweight speaker adapter.

## 8. Streaming distillation

    python scripts/train.py \
      --stage streaming-distill \
      --config configs/training/streaming_distill.yaml \
      --model-config configs/models/hdvr.yaml \
      --train-manifest data/processed/train_features.jsonl \
      --validation-manifest data/processed/validation_features.jsonl \
      --output-dir checkpoints/hdvr/streaming \
      --base-checkpoint checkpoints/adapters/target_speaker/best.pt \
      --device cuda

Use the resulting student checkpoint for low-latency HDVR planning.

## 9. Prepare Twin Converter training pairs

The universal converter requires parallel rows containing:

    source_audio,target_audio

source_audio is carrier speech from any TTS model.
target_audio is the consented target speaker delivering the same or closely matched content.

Prepare conversion features:

    python scripts/prepare_conversion_dataset.py \
      --input-manifest data/manifests/conversion_train.jsonl \
      --output-manifest data/processed/conversion_train_features.jsonl \
      --feature-dir data/processed/conversion/train \
      --device cuda

Repeat for validation.

## 10. Train the universal Twin Converter

    python scripts/train.py \
      --stage twin-converter \
      --config configs/training/twin_converter.yaml \
      --model-config configs/models/hdvr.yaml \
      --train-manifest data/processed/conversion_train_features.jsonl \
      --validation-manifest data/processed/conversion_validation_features.jsonl \
      --output-dir checkpoints/converter/target_speaker \
      --device cuda

This checkpoint lets non-native TTS providers become target Digital Twin output.

## 11. Configure runtime

For Gemini-native target-voice output, create the provider binding:

    python scripts/register_gemini_voice.py \
      --voice-id target_speaker \
      --source-audio target_reference.wav \
      --consent-audio target_consent.wav \
      --display-name "BRUTUS Digital Twin"

For generic providers with local conversion:

    export BVT_TWIN_CONVERTER_CHECKPOINT=checkpoints/converter/target_speaker/best.pt
    export BVT_CONVERTER_DEVICE=cuda

Set Gemini as default:

    export GEMINI_API_KEY=...
    export BVT_DEFAULT_TTS_PROVIDER=gemini

Start the API:

    uvicorn voice_twin.api.server:app --host 0.0.0.0 --port 8787

## 12. Synthesis test

    python scripts/synthesize.py \
      --voice target_speaker \
      --text "BRUTUS Digital Twin is online." \
      --output artifacts/generated_audio/smoke.wav

Every public output must contain the Digital Twin invariant.

## 13. Evaluation

Clone evaluation:

    python scripts/evaluate.py \
      --suite cloning \
      --reference data/eval_sets/neutral/real.wav \
      --generated artifacts/generated_audio/smoke.wav \
      --self-reference data/eval_sets/neutral/real_second_take.wav \
      --reference-text "BRUTUS Digital Twin is online." \
      --hypothesis-text "BRUTUS Digital Twin is online." \
      --heavy-identity \
      --device cuda \
      --output artifacts/reports/clone.json

Performance evaluation:

    python scripts/evaluate.py \
      --suite performance \
      --reference data/eval_sets/performance_copy/reference.wav \
      --generated artifacts/generated_audio/performance.wav \
      --output artifacts/reports/performance.json

Accent:

    python scripts/evaluate.py \
      --suite accent \
      --reference data/eval_sets/accent_heavy/reference.wav \
      --generated artifacts/generated_audio/accent.wav \
      --output artifacts/reports/accent.json

Robot over-air:

    python scripts/evaluate.py \
      --suite robot \
      --reference artifacts/generated_audio/smoke.wav \
      --generated data/eval_sets/robot/measurement_mic.wav \
      --output artifacts/reports/robot.json

Use --require-all-gates only on a suite that actually computes all configured gate metrics.

## 14. Resume training

Every stage supports:

    --resume checkpoints/.../latest.pt

Use the same config, manifests, output directory, and base checkpoint when resuming.

## 15. Release acceptance

Do not claim a voice model is finished because training loss converged.

Release requires:
- ECAPA and SSL identity retention against human self-similarity
- WER/CER
- F0 correlation and log-F0 RMSE
- pause precision/recall/F1 and timing errors
- duration error
- breath-event accuracy
- accent/allophone tests
- TTFA and real-time factor
- robot over-air evaluation
- blinded speaker ABX
- blinded real-vs-synthetic study

Engineering thresholds live in configs/evaluation/success_gates.yaml.

They are starting gates, not universal scientific constants.
