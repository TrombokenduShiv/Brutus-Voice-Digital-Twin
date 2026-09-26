from __future__ import annotations

import argparse
from time import perf_counter


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--audio-seconds",type=float,required=True)
    p.add_argument("--compute-seconds",type=float,required=True)
    args=p.parse_args()
    print({"realtime_factor":args.compute_seconds/args.audio_seconds})


if __name__=="__main__":
    main()
