from __future__ import annotations

import argparse
import yaml


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--stage",required=True,choices=["hdvr","speaker-adapt","accent","prosody","events","streaming-distill"])
    p.add_argument("--config",required=True)
    args=p.parse_args()
    with open(args.config,"r",encoding="utf-8") as f:
        config=yaml.safe_load(f)
    print({"stage":args.stage,"config_loaded":bool(config)})
    print("Training entrypoint validated. Dataset/model injection is stage-specific; no pretrained weights are fabricated.")


if __name__=="__main__":
    main()
