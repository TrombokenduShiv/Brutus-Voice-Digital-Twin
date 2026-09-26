from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.features import prepare_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract HDVR training tensors from audio manifests.")
    parser.add_argument("--input-manifest", "--input", dest="input_manifest", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--aligner", choices=["ctc", "proportional"], default="ctc")
    parser.add_argument("--ctc-model", default="facebook/wav2vec2-base-960h")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--lightweight",
        action="store_true",
        help="Use deterministic DSP embeddings instead of ECAPA/WavLM. Smoke tests only.",
    )
    parser.add_argument("--memory-slots", type=int, default=64)
    args = parser.parse_args()

    output = prepare_manifest(
        args.input_manifest,
        args.output_manifest,
        args.feature_dir,
        sample_rate=args.sample_rate,
        aligner=args.aligner,
        ctc_model=args.ctc_model,
        device=args.device,
        lightweight=args.lightweight,
        memory_slots=args.memory_slots,
    )
    rows = sum(1 for line in Path(output).read_text(encoding="utf-8").splitlines() if line.strip())
    print(json.dumps({"output_manifest": str(output), "utterances": rows}, indent=2))


if __name__ == "__main__":
    main()
