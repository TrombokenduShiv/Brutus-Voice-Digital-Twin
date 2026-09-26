from __future__ import annotations

import argparse
import json

from training.stages import (
    adapt_speaker,
    train_accent,
    train_conversion,
    train_events,
    train_hdvr,
    train_prosody,
    train_streaming,
)


STAGES = {
    "hdvr": train_hdvr.train,
    "speaker-adapt": adapt_speaker.train,
    "accent": train_accent.train,
    "prosody": train_prosody.train,
    "events": train_events.train,
    "streaming-distill": train_streaming.train,
    "twin-converter": train_conversion.train,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BRUTUS Voice Digital Twin models.")
    parser.add_argument("--stage", required=True, choices=sorted(STAGES))
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-config", default="configs/models/hdvr.yaml")
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--validation-manifest")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--base-checkpoint")
    parser.add_argument("--resume")
    args = parser.parse_args()

    state = STAGES[args.stage](
        config_path=args.config,
        model_config_path=args.model_config,
        train_manifest=args.train_manifest,
        validation_manifest=args.validation_manifest,
        output_dir=args.output_dir,
        device=args.device,
        base_checkpoint=args.base_checkpoint,
        resume=args.resume,
    )
    print(json.dumps({
        "stage": args.stage,
        "step": state.step,
        "epoch": state.epoch,
        "best_metric": state.best_metric,
        "output_dir": args.output_dir,
    }, indent=2))


if __name__ == "__main__":
    main()
