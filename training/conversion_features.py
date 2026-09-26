from __future__ import annotations

import hashlib
import json
from pathlib import Path

import librosa
import numpy as np

from training.features import ACCENT_SCALE, _fallback_embedding
from voice_twin.accent.acoustic_features import extract_accent_features
from voice_twin.audio.io import load_audio
from voice_twin.audio.normalize import peak_normalize
from voice_twin.audio.resample import resample
from voice_twin.events.breath_detector import detect_breaths
from voice_twin.identity.ecapa_encoder import EcapaSpeakerEncoder
from voice_twin.identity.vocal_profile import vocal_profile
from voice_twin.prosody.energy import extract_energy
from voice_twin.prosody.f0 import extract_f0


def log_mel_100(audio: np.ndarray, sample_rate: int = 24000) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=np.asarray(audio, dtype=np.float32),
        sr=sample_rate,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        n_mels=100,
        fmin=0,
        fmax=12000,
        power=1.0,
    )
    return np.log(np.clip(mel, 1e-5, None)).T.astype(np.float32)


def _dtw_align_target(source_mel: np.ndarray, target_mel: np.ndarray) -> np.ndarray:
    if len(source_mel) == 0 or len(target_mel) == 0:
        raise ValueError("cannot align empty mel spectrograms")
    x = source_mel.T
    y = target_mel.T
    _, path = librosa.sequence.dtw(X=x, Y=y, metric="cosine")
    path = path[::-1]
    mapping: dict[int, list[int]] = {}
    for source_i, target_i in path:
        mapping.setdefault(int(source_i), []).append(int(target_i))
    aligned = []
    last = 0
    for i in range(len(source_mel)):
        indices = mapping.get(i)
        if indices:
            last = int(round(float(np.mean(indices))))
        last = min(max(last, 0), len(target_mel) - 1)
        aligned.append(target_mel[last])
    return np.stack(aligned).astype(np.float32)


def _style_vector(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    f0 = extract_f0(audio, sample_rate)
    voiced = f0[f0 > 0]
    energy = extract_energy(audio, sample_rate)
    breaths = detect_breaths(audio, sample_rate)
    duration = len(audio) / sample_rate
    prosody = np.asarray(
        [
            (float(voiced.mean()) / 500.0) if len(voiced) else 0.0,
            min(1.0, float(energy.mean()) / 0.10) if len(energy) else 0.0,
            1.0,
            float(len(voiced) / max(1, len(f0))),
            min(1.0, len(breaths) / max(duration, 1e-3) / 2.0),
        ],
        dtype=np.float32,
    )
    accent = extract_accent_features(audio, sample_rate) / ACCENT_SCALE
    event = np.zeros(7, dtype=np.float32)
    event[0] = 1.0
    if breaths:
        event[0] = 0.0
        event[1] = 1.0
    return np.concatenate([prosody, accent, event]).astype(np.float32)


def prepare_conversion_manifest(
    input_manifest: str | Path,
    output_manifest: str | Path,
    feature_dir: str | Path,
    *,
    sample_rate: int = 24000,
    device: str = "cpu",
    lightweight: bool = False,
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

    with output_manifest.open("w", encoding="utf-8") as out:
        for index, row in enumerate(rows):
            if "source_audio" not in row or "target_audio" not in row:
                raise ValueError("conversion rows require source_audio and target_audio")
            source, ssr = load_audio(row["source_audio"])
            target, tsr = load_audio(row["target_audio"])
            source = peak_normalize(resample(source, ssr, sample_rate))
            target = peak_normalize(resample(target, tsr, sample_rate))
            source_mel = log_mel_100(source, sample_rate)
            target_mel = _dtw_align_target(source_mel, log_mel_100(target, sample_rate))

            if lightweight:
                identity = _fallback_embedding(target, sample_rate, 192)
            else:
                identity = speaker_encoder.encode(target, sample_rate)
            vocal = vocal_profile(
                extract_f0(target, sample_rate),
                extract_energy(target, sample_rate),
            )
            style = _style_vector(target, sample_rate)
            utterance_id = str(
                row.get("id")
                or hashlib.sha1(
                    f"{row['source_audio']}|{row['target_audio']}".encode()
                ).hexdigest()[:12]
            )
            feature_path = feature_dir / f"{utterance_id}.npz"
            np.savez_compressed(
                feature_path,
                source_mel=source_mel,
                target_mel=target_mel,
                identity=identity.astype(np.float32),
                vocal=vocal.astype(np.float32),
                style=style,
            )
            output_row = dict(row)
            output_row["id"] = utterance_id
            output_row["duration"] = len(source) / sample_rate
            output_row["feature_path"] = str(feature_path.resolve())
            out.write(json.dumps(output_row, ensure_ascii=False) + "\n")
    return output_manifest
