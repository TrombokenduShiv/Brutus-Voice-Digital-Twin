from __future__ import annotations

import argparse

from voice_twin.acoustic.qwen_backend import QwenVoiceCloneBackend
from voice_twin.audio.io import save_audio


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--text",required=True)
    p.add_argument("--reference",required=True)
    p.add_argument("--reference-text")
    p.add_argument("--language",default="English")
    p.add_argument("--output",default="artifacts/generated_audio/output.wav")
    args=p.parse_args()
    backend=QwenVoiceCloneBackend()
    generated=backend.synthesize(args.text,args.language,ref_audio=args.reference,ref_text=args.reference_text)
    save_audio(args.output,generated.waveform,generated.sample_rate)
    print(args.output)


if __name__=="__main__":
    main()
