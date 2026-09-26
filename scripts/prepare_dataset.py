from __future__ import annotations

import argparse
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args()
    Path(args.output).mkdir(parents=True,exist_ok=True)
    print(f"Dataset preparation root ready: {args.output}. Add transcripts before feature extraction.")


if __name__=="__main__":
    main()
