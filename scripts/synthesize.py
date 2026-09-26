from __future__ import annotations

import argparse

from voice_twin.api.runtime import get_engine
from voice_twin.audio.io import save_audio
from voice_twin.schemas import SynthesisRequest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--voice", required=True, help="Enrolled local VoiceDNA profile id")
    parser.add_argument("--text", required=True)
    parser.add_argument("--provider", default=None, help="Defaults to Gemini")
    parser.add_argument("--language", default="English")
    parser.add_argument("--emotion")
    parser.add_argument("--output", default="artifacts/generated_audio/output.wav")
    args = parser.parse_args()

    generated = get_engine().synthesize(
        SynthesisRequest(
            text=args.text,
            voice_id=args.voice,
            provider=args.provider,
            language=args.language,
            emotion=args.emotion,
        )
    )
    if not generated.digital_twin:
        raise RuntimeError("refusing to write non-digital-twin audio")
    save_audio(args.output, generated.waveform, generated.sample_rate)
    print({
        "output": args.output,
        "provider": generated.provider,
        "digital_twin": generated.digital_twin,
    })


if __name__ == "__main__":
    main()
