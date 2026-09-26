from __future__ import annotations

from voice_twin.acoustic.base import AcousticBackend, GeneratedAudio
from voice_twin.audio.resample import resample
from voice_twin.device.robot_renderer import RobotRenderer
from voice_twin.schemas import SynthesisRequest
from voice_twin.streaming.pcm_stream import audio_frames


class VoiceTwinEngine:
    def __init__(self, backend: AcousticBackend, renderer: RobotRenderer | None = None):
        self.backend = backend
        self.renderer = renderer or RobotRenderer()

    def synthesize(self, request: SynthesisRequest, *, ref_audio=None, ref_text: str | None = None) -> GeneratedAudio:
        generated = self.backend.synthesize(
            text=request.text,
            language=request.language,
            ref_audio=ref_audio or request.reference_audio,
            ref_text=ref_text or request.reference_text,
        )
        waveform = self.renderer.render(generated.waveform, generated.sample_rate)
        return GeneratedAudio(waveform, self.renderer.target_sr)

    def stream(self, request: SynthesisRequest, *, ref_audio=None, ref_text: str | None = None):
        generated = self.synthesize(request, ref_audio=ref_audio, ref_text=ref_text)
        return audio_frames(generated.waveform, generated.sample_rate)
