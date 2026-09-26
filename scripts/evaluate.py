from __future__ import annotations

import argparse
import json

from evaluation.metrics.f0 import f0_metrics


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--reference-f0")
    p.add_argument("--predicted-f0")
    args=p.parse_args()
    if not args.reference_f0 or not args.predicted_f0:
        print(json.dumps({"status":"ready","message":"Provide extracted F0 arrays or run the evaluation suite."}))
        return
    import numpy as np
    print(json.dumps(f0_metrics(np.load(args.reference_f0),np.load(args.predicted_f0)),indent=2))


if __name__=="__main__":
    main()
