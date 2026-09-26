from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_MODELS = {
    "ctc": "facebook/wav2vec2-base-960h",
    "ssl": "microsoft/wavlm-base-plus-sv",
    "ecapa": "speechbrain/spkrec-ecapa-voxceleb",
    "bigvgan": "nvidia/bigvgan_v2_24khz_100band_256x",
    "qwen": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
}


def _snapshot(model_id: str, cache_dir: Path | None) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=model_id,
        cache_dir=str(cache_dir) if cache_dir else None,
        local_files_only=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prefetch BRUTUS Voice Digital Twin model dependencies."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(DEFAULT_MODELS),
        default=["ctc", "ssl", "ecapa"],
        help="Named model groups to prefetch.",
    )
    parser.add_argument("--cache-dir")
    parser.add_argument(
        "--include-qwen",
        action="store_true",
        help="Also download the optional Qwen local TTS baseline.",
    )
    parser.add_argument(
        "--include-vocoder",
        action="store_true",
        help="Also download BigVGAN for the local Twin Converter.",
    )
    args = parser.parse_args()

    selected = list(dict.fromkeys(args.models))
    if args.include_qwen and "qwen" not in selected:
        selected.append("qwen")
    if args.include_vocoder and "bigvgan" not in selected:
        selected.append("bigvgan")

    cache_dir = Path(args.cache_dir).expanduser().resolve() if args.cache_dir else None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)

    resolved: dict[str, dict[str, str]] = {}
    for name in selected:
        model_id = DEFAULT_MODELS[name]
        print(f"[download] {name}: {model_id}")
        path = _snapshot(model_id, cache_dir)
        resolved[name] = {"model_id": model_id, "path": path}
        print(f"[ready]    {path}")

    print(json.dumps(resolved, indent=2))


if __name__ == "__main__":
    main()
