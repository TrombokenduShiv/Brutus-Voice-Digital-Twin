from __future__ import annotations

import librosa
import numpy as np


def mel_spectral_distance(
    reference: np.ndarray,
    generated: np.ndarray,
    sample_rate: int,
    n_mels: int = 80,
) -> dict[str, float]:
    def mel(x):
        m = librosa.feature.melspectrogram(
            y=np.asarray(x, dtype=np.float32),
            sr=sample_rate,
            n_fft=1024,
            hop_length=256,
            n_mels=n_mels,
            fmax=min(12000, sample_rate // 2),
        )
        return librosa.power_to_db(m + 1e-10, ref=np.max)

    r, g = mel(reference), mel(generated)
    frames = min(r.shape[1], g.shape[1])
    if frames == 0:
        return {"mel_db_rmse": float("inf"), "mel_db_mae": float("inf")}
    r, g = r[:, :frames], g[:, :frames]
    diff = r - g
    return {
        "mel_db_rmse": float(np.sqrt(np.mean(diff * diff))),
        "mel_db_mae": float(np.mean(np.abs(diff))),
    }
