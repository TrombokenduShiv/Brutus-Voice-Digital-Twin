from __future__ import annotations

import numpy as np

from voice_twin.vocoder.base import Vocoder


class BigVGANVocoder(Vocoder):
    """Lazy wrapper around NVIDIA's official BigVGAN Hugging Face integration."""

    def __init__(
        self,
        model_id: str = "nvidia/bigvgan_v2_24khz_100band_256x",
        device: str = "cuda",
        use_cuda_kernel: bool = False,
    ):
        self.model_id = model_id
        self.device = device
        self.use_cuda_kernel = use_cuda_kernel
        self._model = None

    def _load(self):
        if self._model is None:
            import bigvgan

            model = bigvgan.BigVGAN.from_pretrained(
                self.model_id,
                use_cuda_kernel=self.use_cuda_kernel,
            )
            model.remove_weight_norm()
            self._model = model.eval().to(self.device)
        return self._model

    @property
    def sample_rate(self) -> int:
        return int(self._load().h.sampling_rate)

    def decode(self, features: np.ndarray) -> np.ndarray:
        import torch

        mel = np.asarray(features, dtype=np.float32)
        if mel.ndim == 2:
            mel = mel[None, ...]
        if mel.ndim != 3:
            raise ValueError("mel must be [T,M] or [B,T,M]")
        tensor = torch.from_numpy(mel).to(self.device).transpose(1, 2)
        with torch.inference_mode():
            waveform = self._load()(tensor)
        return waveform[0, 0].detach().cpu().numpy().astype(np.float32)
