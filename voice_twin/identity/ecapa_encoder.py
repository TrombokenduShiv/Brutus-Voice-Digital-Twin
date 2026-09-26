from __future__ import annotations

from pathlib import Path

import numpy as np


class EcapaSpeakerEncoder:
    def __init__(self, source: str = "speechbrain/spkrec-ecapa-voxceleb", savedir: str = "models/pretrained/ecapa"):
        self.source = source
        self.savedir = savedir
        self._model = None

    def _load(self):
        if self._model is None:
            from speechbrain.inference.speaker import EncoderClassifier
            Path(self.savedir).mkdir(parents=True, exist_ok=True)
            self._model = EncoderClassifier.from_hparams(source=self.source, savedir=self.savedir)
        return self._model

    def encode(self, waveform: np.ndarray) -> np.ndarray:
        import torch
        wav = torch.from_numpy(np.asarray(waveform, dtype=np.float32)).unsqueeze(0)
        emb = self._load().encode_batch(wav).detach().cpu().numpy().reshape(-1)
        return emb.astype(np.float32)
