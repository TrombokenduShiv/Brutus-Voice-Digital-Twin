from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

from voice_twin.accent.acoustic_features import extract_accent_features
from voice_twin.audio.io import load_audio
from voice_twin.audio.normalize import peak_normalize
from voice_twin.audio.resample import resample
from voice_twin.enrollment.alignment import CTCForcedAligner, ProportionalAligner, TokenAlignment
from voice_twin.events.breath_detector import detect_breaths
from voice_twin.frontend.phonemizer import stable_phone_id
from voice_twin.identity.ecapa_encoder import EcapaSpeakerEncoder
from voice_twin.identity.ssl_encoder import SSLSpeakerEncoder
from voice_twin.identity.vocal_profile import vocal_profile
from voice_twin.prosody.energy import extract_energy
from voice_twin.prosody.f0 import extract_f0


ACCENT_SCALE = np.asarray(
    [500.0, 500.0, 500.0, 0.20, 8000.0, 0.50, 1500.0, 4000.0],
    dtype=np.float32,
)


@dataclass
class PreparedUtterance:
    row: dict
    token_ids: np.ndarray
    identity: np.ndarray
    vocal: np.ndarray
    ssl_target: np.ndarray
    prosody_target: np.ndarray
    accent_target: np.ndarray
    acoustic_target: np.ndarray
    duration_ms: np.ndarray
    pause_type: np.ndarray
    pause_duration_ms: np.ndarray
    event_target: np.ndarray
    channel_id: int


def _fallback_embedding(audio: np.ndarray, sample_rate: int, dim: int) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=np.asarray(audio, dtype=np.float32),
        sr=sample_rate,
        n_mels=64,
        fmax=min(8000, sample_rate // 2),
    )
    stats = np.concatenate(
        [np.log1p(mel).mean(axis=1), np.log1p(mel).std(axis=1)]
    )
    source = np.linspace(0.0, 1.0, len(stats))
    target = np.linspace(0.0, 1.0, dim)
    emb = np.interp(target, source, stats).astype(np.float32)
    emb -= emb.mean()
    return emb / max(float(np.linalg.norm(emb)), 1e-8)


def _mel_target(segment: np.ndarray, sample_rate: int, bins: int = 80) -> np.ndarray:
    x = np.asarray(segment, dtype=np.float32)
    if len(x) < 256:
        x = np.pad(x, (0, 256 - len(x)))
    mel = librosa.feature.melspectrogram(
        y=x,
        sr=sample_rate,
        n_fft=1024,
        hop_length=256,
        n_mels=bins,
        fmax=min(12000, sample_rate // 2),
        power=2.0,
    )
    db = librosa.power_to_db(mel + 1e-10, ref=np.max)
    return ((db.mean(axis=1) + 80.0) / 80.0).clip(0.0, 1.0).astype(np.float32)


def _pause_class(duration_ms: float) -> int:
    if duration_ms < 30:
        return 0
    if duration_ms < 150:
        return 1
    if duration_ms < 350:
        return 2
    if duration_ms < 700:
        return 3
    return 4


def _channel_id(row: dict, classes: int = 32) -> int:
    value = str(row.get("session_id") or row.get("recording_day") or row.get("id") or "unknown")
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") % classes


def _segment_features(
    audio: np.ndarray,
    sample_rate: int,
    alignment: list[TokenAlignment],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    breaths = detect_breaths(audio, sample_rate)
    accent_rows, acoustic_rows, prosody_rows = [], [], []
    durations, pause_types, pause_durations, event_targets = [], [], [], []

    raw_durations = np.asarray([max(0.001, x.duration_s) for x in alignment], dtype=np.float32)
    median_duration = max(float(np.median(raw_durations)), 1e-3)

    for i, item in enumerate(alignment):
        start = max(0, int(item.start_s * sample_rate))
        end = min(len(audio), max(start + 1, int(item.end_s * sample_rate)))
        segment = audio[start:end]

        accent = extract_accent_features(segment, sample_rate)
        accent_rows.append(accent / ACCENT_SCALE)
        acoustic_rows.append(_mel_target(segment, sample_rate))

        f0 = extract_f0(segment, sample_rate)
        voiced = f0[f0 > 0]
        f0_mean = float(voiced.mean()) if len(voiced) else 0.0
        energy = float(np.sqrt(np.mean(segment * segment) + 1e-12))
        voicing = float(len(voiced) / max(1, len(f0)))
        duration_rate = float(item.duration_s / median_duration)

        next_start = alignment[i + 1].start_s if i + 1 < len(alignment) else item.end_s
        pause_ms = max(0.0, (next_start - item.end_s) * 1000.0)
        pause_types.append(_pause_class(pause_ms))
        pause_durations.append(pause_ms)
        durations.append(item.duration_s * 1000.0)

        gap_end = next_start if i + 1 < len(alignment) else min(
            len(audio) / sample_rate, item.end_s + 0.4
        )
        overlap = [
            b for b in breaths
            if b.start_s < gap_end and b.end_s > item.end_s
        ]
        breath_probability = max((b.confidence for b in overlap), default=0.0)
        event_targets.append(1 if overlap else 0)

        prosody_rows.append(
            [
                f0_mean,
                min(1.0, energy / 0.10),
                duration_rate,
                voicing,
                breath_probability,
            ]
        )

    return (
        np.asarray(accent_rows, dtype=np.float32),
        np.asarray(acoustic_rows, dtype=np.float32),
        np.asarray(prosody_rows, dtype=np.float32),
        np.asarray(durations, dtype=np.float32),
        np.asarray(pause_types, dtype=np.int64),
        np.asarray(pause_durations, dtype=np.float32),
        np.asarray(event_targets, dtype=np.int64),
    )


def _alignment(
    audio: np.ndarray,
    sample_rate: int,
    row: dict,
    aligner: str,
    ctc_model: str,
    device: str,
) -> list[TokenAlignment]:
    language = str(row.get("language", "en-us"))
    if aligner == "ctc":
        return CTCForcedAligner(ctc_model, device).align(
            audio, sample_rate, str(row["text"]), language
        )
    return ProportionalAligner().align(audio, sample_rate, str(row["text"]), language)


def extract_utterance(
    row: dict,
    *,
    sample_rate: int = 24000,
    aligner: str = "ctc",
    ctc_model: str = "facebook/wav2vec2-base-960h",
    device: str = "cpu",
    speaker_encoder: EcapaSpeakerEncoder | None = None,
    ssl_encoder: SSLSpeakerEncoder | None = None,
    lightweight: bool = False,
) -> PreparedUtterance:
    audio, source_sr = load_audio(row["audio"])
    audio = peak_normalize(resample(audio, source_sr, sample_rate))
    alignment = _alignment(audio, sample_rate, row, aligner, ctc_model, device)
    if not alignment:
        raise ValueError(f"no alignment produced for {row.get('id', row['audio'])}")

    token_ids = np.asarray(
        [stable_phone_id(item.token) for item in alignment],
        dtype=np.int64,
    )
    f0 = extract_f0(audio, sample_rate)
    energy = extract_energy(audio, sample_rate)
    vocal = vocal_profile(f0, energy)

    if lightweight:
        identity = _fallback_embedding(audio, sample_rate, 192)
        ssl_target = _fallback_embedding(audio, sample_rate, 768)
    else:
        speaker_encoder = speaker_encoder or EcapaSpeakerEncoder(device=device)
        ssl_encoder = ssl_encoder or SSLSpeakerEncoder(device=device)
        identity = speaker_encoder.encode(audio, sample_rate)
        ssl_target = ssl_encoder.encode(audio, sample_rate)

    (
        accent_target,
        acoustic_target,
        prosody_target,
        duration_ms,
        pause_type,
        pause_duration_ms,
        event_target,
    ) = _segment_features(audio, sample_rate, alignment)

    row = dict(row)
    row["duration"] = len(audio) / sample_rate
    return PreparedUtterance(
        row=row,
        token_ids=token_ids,
        identity=identity.astype(np.float32),
        vocal=vocal.astype(np.float32),
        ssl_target=ssl_target.astype(np.float32),
        prosody_target=prosody_target,
        accent_target=accent_target,
        acoustic_target=acoustic_target,
        duration_ms=duration_ms,
        pause_type=pause_type,
        pause_duration_ms=pause_duration_ms,
        event_target=event_target,
        channel_id=_channel_id(row),
    )


def _fixed_memory(vectors: list[np.ndarray], slots: int = 64) -> np.ndarray:
    x = np.stack(vectors).astype(np.float32)
    if len(x) >= slots:
        idx = np.linspace(0, len(x) - 1, slots).round().astype(int)
        return x[idx]
    reps = int(np.ceil(slots / len(x)))
    return np.tile(x, (reps, 1))[:slots]


def prepare_manifest(
    input_manifest: str | Path,
    output_manifest: str | Path,
    feature_dir: str | Path,
    *,
    sample_rate: int = 24000,
    aligner: str = "ctc",
    ctc_model: str = "facebook/wav2vec2-base-960h",
    device: str = "cpu",
    lightweight: bool = False,
    memory_slots: int = 64,
) -> Path:
    input_manifest = Path(input_manifest)
    output_manifest = Path(output_manifest)
    feature_dir = Path(feature_dir)
    feature_dir.mkdir(parents=True, exist_ok=True)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)

    rows = [
        json.loads(line)
        for line in input_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    speaker_encoder = None if lightweight else EcapaSpeakerEncoder(device=device)
    ssl_encoder = None if lightweight else SSLSpeakerEncoder(device=device)

    prepared: list[PreparedUtterance] = []
    for row in rows:
        prepared.append(
            extract_utterance(
                row,
                sample_rate=sample_rate,
                aligner=aligner,
                ctc_model=ctc_model,
                device=device,
                speaker_encoder=speaker_encoder,
                ssl_encoder=ssl_encoder,
                lightweight=lightweight,
            )
        )

    by_speaker: dict[str, list[PreparedUtterance]] = defaultdict(list)
    for item in prepared:
        by_speaker[str(item.row["speaker_id"])].append(item)

    speaker_memory: dict[str, np.ndarray] = {}
    prosody_prior: dict[str, np.ndarray] = {}
    event_prior: dict[str, np.ndarray] = {}
    accent_prior: dict[str, dict[int, np.ndarray]] = {}

    for speaker, items in by_speaker.items():
        speaker_memory[speaker] = _fixed_memory(
            [x.identity for x in items],
            slots=memory_slots,
        )
        all_prosody = np.concatenate([x.prosody_target for x in items], axis=0)
        prosody_prior[speaker] = all_prosody.mean(axis=0).astype(np.float32)

        events = np.concatenate([x.event_target for x in items], axis=0)
        counts = np.bincount(events, minlength=7).astype(np.float32)
        event_prior[speaker] = counts / max(float(counts.sum()), 1.0)

        accum: dict[int, list[np.ndarray]] = defaultdict(list)
        for item in items:
            for token_id, feat in zip(item.token_ids.tolist(), item.accent_target):
                accum[int(token_id)].append(feat)
        accent_prior[speaker] = {
            token: np.stack(values).mean(axis=0).astype(np.float32)
            for token, values in accum.items()
        }

    with output_manifest.open("w", encoding="utf-8") as out:
        for item in prepared:
            speaker = str(item.row["speaker_id"])
            length = len(item.token_ids)
            accent_context = np.stack(
                [
                    accent_prior[speaker].get(int(token), np.zeros(8, dtype=np.float32))
                    for token in item.token_ids
                ]
            )
            prosody_context = np.tile(prosody_prior[speaker], (length, 1))
            event_context = np.tile(event_prior[speaker], (length, 1))

            utterance_id = str(item.row.get("id") or hashlib.sha1(
                str(item.row["audio"]).encode()
            ).hexdigest()[:12])
            feature_path = feature_dir / f"{utterance_id}.npz"
            np.savez_compressed(
                feature_path,
                token_ids=item.token_ids,
                identity=item.identity,
                vocal=item.vocal,
                speaker_memory=speaker_memory[speaker],
                accent_context=accent_context.astype(np.float32),
                prosody_context=prosody_context.astype(np.float32),
                prosody_target=item.prosody_target,
                event_context=event_context.astype(np.float32),
                acoustic_target=item.acoustic_target,
                speaker_target=item.identity,
                ssl_target=item.ssl_target,
                duration_ms=item.duration_ms,
                pause_type=item.pause_type,
                pause_duration_ms=item.pause_duration_ms,
                accent_target=item.accent_target,
                event_target=item.event_target,
                channel_id=np.asarray(item.channel_id, dtype=np.int64),
            )
            output_row = dict(item.row)
            output_row["id"] = utterance_id
            output_row["feature_path"] = str(feature_path.resolve())
            out.write(json.dumps(output_row, ensure_ascii=False) + "\n")
    return output_manifest
