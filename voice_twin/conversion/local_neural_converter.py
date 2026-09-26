from __future__ import annotations

from pathlib import Path

import numpy as np

from training.conversion_features import log_mel_100
from voice_twin.acoustic.base import GeneratedAudio
from voice_twin.conversion.neural_converter import TwinMelConverter
from voice_twin.conversion.twin_converter import TwinConversionBackend
from voice_twin.hdvr.conditioning import DigitalTwinPlan
from voice_twin.profiles.voice_dna import VoiceDNA
from voice_twin.vocoder.bigvgan import BigVGANVocoder


_ACCENT_SCALE = np.asarray(
    [500.0, 500.0, 500.0, 0.20, 8000.0, 0.50, 1500.0, 4000.0],
    dtype=np.float32,
)


def _fit_vector(values, dim: int) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32).reshape(-1)
    if len(x) >= dim:
        return x[:dim]
    return np.pad(x, (0, dim - len(x)))


def style_vector_from_profile(profile: VoiceDNA) -> np.ndarray:
    style = profile.style_profile or {}
    f0 = float(style.get("f0_mean_hz", 0.0)) / 500.0
    energy = min(1.0, float(style.get("energy_mean", 0.0)) / 0.10)
    pace = float(style.get("pace_ratio", 1.0))
    voicing = float(style.get("voicing_ratio", 0.8))
    breath = float(style.get("breath_probability", 0.0))
    prosody = np.asarray([f0, energy, pace, voicing, breath], dtype=np.float32)

    accent_rows = []
    for cell in (profile.accent_atlas or {}).values():
        if isinstance(cell, dict) and isinstance(cell.get("mean"), list):
            value = np.asarray(cell["mean"][:8], dtype=np.float32)
            if len(value) == 8:
                accent_rows.append(value / _ACCENT_SCALE)
    accent = (
        np.mean(np.stack(accent_rows), axis=0).astype(np.float32)
        if accent_rows
        else np.zeros(8, dtype=np.float32)
    )

    event = np.zeros(7, dtype=np.float32)
    event[0] = 1.0
    signature = profile.event_signature or {}
    distribution = signature.get("distribution") if isinstance(signature, dict) else None
    if isinstance(distribution, list) and len(distribution) >= 7:
        event = np.asarray(distribution[:7], dtype=np.float32)
        event /= max(float(event.sum()), 1e-8)
    elif signature.get("breath_probability") is not None:
        p = float(signature["breath_probability"])
        event[0], event[1] = max(0.0, 1.0 - p), min(1.0, p)

    return np.concatenate([prosody, accent, event]).astype(np.float32)


class LocalNeuralTwinConverter(TwinConversionBackend):
    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device: str = "cuda",
        vocoder_model: str = "nvidia/bigvgan_v2_24khz_100band_256x",
        use_cuda_kernel: bool = False,
    ):
        self.checkpoint = str(checkpoint)
        self.device = device
        self.vocoder = BigVGANVocoder(
            vocoder_model,
            device=device,
            use_cuda_kernel=use_cuda_kernel,
        )
        self._model = None

    def _load_model(self):
        if self._model is None:
            import torch

            payload = torch.load(
                self.checkpoint,
                map_location=self.device,
                weights_only=False,
            )
            config = payload.get("model_config", {})
            model = TwinMelConverter(**config)
            state = payload.get("export_model_state") or payload["model_state"]
            model.load_state_dict(state, strict=False)
            self._model = model.eval().to(self.device)
        return self._model

    def convert(
        self,
        generated: GeneratedAudio,
        profile: VoiceDNA,
        plan: DigitalTwinPlan,
    ) -> GeneratedAudio:
        import torch

        source_mel = log_mel_100(generated.waveform, generated.sample_rate)
        identity = _fit_vector(profile.identity_core, 192)
        vocal = _fit_vector(profile.vocal_profile, 128)
        style = style_vector_from_profile(profile)

        model = self._load_model()
        with torch.inference_mode():
            result = model(
                torch.from_numpy(source_mel).unsqueeze(0).to(self.device),
                torch.from_numpy(identity).unsqueeze(0).to(self.device),
                torch.from_numpy(vocal).unsqueeze(0).to(self.device),
                torch.from_numpy(style).unsqueeze(0).to(self.device),
                torch.ones(1, len(source_mel), dtype=torch.bool, device=self.device),
                causal=True,
            )
        mel = result["mel"][0].detach().cpu().numpy()
        waveform = self.vocoder.decode(mel)
        return GeneratedAudio(
            waveform=waveform,
            sample_rate=self.vocoder.sample_rate,
            provider=generated.provider,
            provider_voice_id=generated.provider_voice_id,
            native_digital_twin=False,
            digital_twin=True,
            metadata={
                **generated.metadata,
                "conversion_backend": "local-neural",
                "converter_checkpoint": self.checkpoint,
                "digital_twin_plan": plan.plan_id,
            },
        )
