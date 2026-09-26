from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import shutil
import sys
from pathlib import Path


def _module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _torch_status() -> dict:
    if not _module("torch"):
        return {"installed": False}
    import torch

    status = {
        "installed": True,
        "version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": bool(
            getattr(torch.backends, "mps", None)
            and torch.backends.mps.is_available()
        ),
    }
    if torch.cuda.is_available():
        status["cuda_version"] = torch.version.cuda
        status["gpu_count"] = torch.cuda.device_count()
        status["gpus"] = [
            {
                "name": torch.cuda.get_device_name(i),
                "vram_gb": round(
                    torch.cuda.get_device_properties(i).total_memory / 1024**3, 2
                ),
            }
            for i in range(torch.cuda.device_count())
        ]
    return status


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the BRUTUS Voice Digital Twin ML training environment."
    )
    parser.add_argument("--manifest", help="Optional feature manifest to validate.")
    parser.add_argument("--require-gemini", action="store_true")
    parser.add_argument("--require-vocoder", action="store_true")
    args = parser.parse_args()

    result = {
        "python": {
            "version": sys.version.split()[0],
            "supported": (3, 11) <= sys.version_info[:2] < (3, 13),
            "platform": platform.platform(),
        },
        "executables": {
            "ffmpeg": shutil.which("ffmpeg"),
            "espeak": shutil.which("espeak") or shutil.which("espeak-ng"),
            "git": shutil.which("git"),
        },
        "packages": {
            name: _module(name)
            for name in [
                "numpy",
                "scipy",
                "librosa",
                "soundfile",
                "torch",
                "torchaudio",
                "transformers",
                "speechbrain",
                "phonemizer",
                "tensorboard",
                "safetensors",
                "google.genai",
            ]
        },
        "torch": _torch_status(),
        "environment": {
            "GEMINI_API_KEY": bool(os.getenv("GEMINI_API_KEY")),
            "BVT_PROFILE_ROOT": os.getenv("BVT_PROFILE_ROOT", "artifacts/voices"),
            "BVT_TWIN_CONVERTER_CHECKPOINT": bool(
                os.getenv("BVT_TWIN_CONVERTER_CHECKPOINT")
            ),
        },
    }

    errors: list[str] = []
    warnings: list[str] = []

    if not result["python"]["supported"]:
        errors.append("Use Python 3.11 or 3.12.")
    for package in ["numpy", "librosa", "soundfile", "torch", "transformers"]:
        if not result["packages"][package]:
            errors.append(f"Missing required package: {package}")
    if not result["executables"]["espeak"]:
        warnings.append(
            "eSpeak/eSpeak-NG not found: phonemizer will use the grapheme fallback."
        )
    if not result["executables"]["ffmpeg"]:
        warnings.append("ffmpeg not found: some audio conversion workflows may fail.")
    if args.require_gemini and not result["environment"]["GEMINI_API_KEY"]:
        errors.append("GEMINI_API_KEY is required for Gemini production synthesis.")
    if args.require_vocoder and not _module("bigvgan"):
        errors.append("BigVGAN is required for local Twin Converter waveform synthesis.")

    if args.manifest:
        manifest = Path(args.manifest)
        if not manifest.exists():
            errors.append(f"Manifest does not exist: {manifest}")
        else:
            rows = [
                json.loads(line)
                for line in manifest.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            missing_features = []
            for row in rows:
                feature = row.get("feature_path")
                if not feature or not Path(feature).exists():
                    missing_features.append(row.get("id", feature))
            result["manifest"] = {
                "path": str(manifest),
                "rows": len(rows),
                "missing_feature_files": missing_features[:20],
            }
            if missing_features:
                errors.append(
                    f"{len(missing_features)} manifest rows have missing feature files."
                )

    result["warnings"] = warnings
    result["errors"] = errors
    result["ok"] = not errors
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["ok"] else 2)


if __name__ == "__main__":
    main()
