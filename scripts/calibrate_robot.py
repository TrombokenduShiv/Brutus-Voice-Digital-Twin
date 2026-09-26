from __future__ import annotations

import argparse
from voice_twin.audio.io import save_audio
from voice_twin.device.calibration import logarithmic_sweep


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",default="artifacts/device_profiles/calibration_sweep.wav")
    p.add_argument("--sample-rate",type=int,default=24000)
    args=p.parse_args()
    save_audio(args.output,logarithmic_sweep(args.sample_rate),args.sample_rate)
    print(args.output)


if __name__=="__main__":
    main()
