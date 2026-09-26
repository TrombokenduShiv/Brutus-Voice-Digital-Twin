from __future__ import annotations

from pathlib import Path

import numpy as np

from voice_twin.audio.resample import resample


class EcapaSpeakerEncoder:
    def __init__(
        self,
        source: str = "speechbrain/spkrec-ecapa-voxceleb",
        savedir: str = "models/pretrained/ecapa",
        device: str = "cpu",
    ):
        self.source = source
        self.savedir = savedir
        self.device = device
        self._model = None

    def _load(self):
        if self._model is None:
            from speechbrain.inference.speaker import EncoderClassifier

            Path(self.savedir).mkdir(parents=True, exist_ok=True)
            self._model = EncoderClassifier.from_hparams(
                source=self.source,
                savedir=self.savedir,
                run_opts={"device": self.device},
            )
        return self._model

    def encode(self, waveform: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        import torch

        x = resample(np.asarray(waveform, dtype=np.float32), sample_rate, 16000)
        wav = torch.from_numpy(x).unsqueeze(0)
        emb = self._load().encode_batch(wav).detach().cpu().numpy().reshape(-1)
        emb = emb / max(float(np.linalg.norm(emb)), 1e-8)
        return emb.astype(np.float32)
