from __future__ import annotations

import numpy as np

from voice_twin.audio.resample import resample


class SSLSpeakerEncoder:
    def __init__(
        self,
        model_id: str = "microsoft/wavlm-base-plus-sv",
        device: str = "cpu",
    ):
        self.model_id = model_id
        self.device = device
        self._extractor = None
        self._model = None

    def _load(self):
        if self._model is None:
            from transformers import AutoFeatureExtractor, AutoModel
            self._extractor = AutoFeatureExtractor.from_pretrained(self.model_id)
            self._model = AutoModel.from_pretrained(self.model_id).to(self.device).eval()
        return self._extractor, self._model

    def encode(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        import torch

        extractor, model = self._load()
        x = resample(np.asarray(waveform, dtype=np.float32), sample_rate, 16000)
        inputs = extractor(x, sampling_rate=16000, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.inference_mode():
            hidden = model(**inputs).last_hidden_state
            embedding = hidden.mean(dim=1)[0]
            embedding = torch.nn.functional.normalize(embedding, dim=0)
        return embedding.cpu().numpy().astype(np.float32)
