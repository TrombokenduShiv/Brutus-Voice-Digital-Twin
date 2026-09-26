from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.conversion_features import prepare_conversion_manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare parallel carrier->target mel pairs for Twin Converter training."
    )
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lightweight", action="store_true")
    args = parser.parse_args()
    path = prepare_conversion_manifest(
        args.input_manifest,
        args.output_manifest,
        args.feature_dir,
        device=args.device,
        lightweight=args.lightweight,
    )
    count = sum(
        1 for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()
    )
    print(json.dumps({"output_manifest": str(path), "pairs": count}, indent=2))


if __name__ == "__main__":
    main()
