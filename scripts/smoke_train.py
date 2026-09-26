from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _feature(path: Path, index: int, tokens: int = 6) -> None:
    rng = np.random.default_rng(1000 + index)
    np.savez_compressed(
        path,
        token_ids=rng.integers(2, 64, size=(tokens,), dtype=np.int64),
        identity=rng.normal(size=(8,)).astype(np.float32),
        vocal=rng.normal(size=(8,)).astype(np.float32),
        speaker_memory=rng.normal(size=(4, 8)).astype(np.float32),
        accent_context=rng.normal(size=(tokens, 8)).astype(np.float32),
        prosody_context=rng.random(size=(tokens, 5)).astype(np.float32),
        prosody_target=np.concatenate(
            [
                rng.uniform(80, 220, size=(tokens, 1)),
                rng.random(size=(tokens, 4)),
            ],
            axis=-1,
        ).astype(np.float32),
        event_context=np.eye(7, dtype=np.float32)[
            np.zeros(tokens, dtype=np.int64)
        ],
        acoustic_target=rng.random(size=(tokens, 10)).astype(np.float32),
        speaker_target=rng.normal(size=(8,)).astype(np.float32),
        ssl_target=rng.normal(size=(12,)).astype(np.float32),
        duration_ms=rng.uniform(30, 180, size=(tokens,)).astype(np.float32),
        pause_type=np.zeros(tokens, dtype=np.int64),
        pause_duration_ms=rng.uniform(0, 80, size=(tokens,)).astype(np.float32),
        accent_target=rng.random(size=(tokens, 8)).astype(np.float32),
        event_target=np.zeros(tokens, dtype=np.int64),
        channel_id=np.asarray(0, dtype=np.int64),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a tiny CPU HDVR train/checkpoint smoke test."
    )
    parser.add_argument("--work-dir", default="artifacts/smoke_train")
    args = parser.parse_args()

    from torch.utils.data import DataLoader

    from training.datasets.collate import collate_features
    from training.datasets.dataset import FeatureDataset
    from training.losses.total import CompositeLoss
    from training.optimizer import build_optimizer
    from training.trainer import Trainer
    from voice_twin.hdvr.model import HDVRModel

    root = Path(args.work_dir)
    features = root / "features"
    features.mkdir(parents=True, exist_ok=True)
    manifest = root / "manifest.jsonl"

    rows = []
    for i in range(4):
        fp = (features / f"sample_{i}.npz").resolve()
        _feature(fp, i)
        rows.append(
            {
                "id": f"smoke_{i}",
                "speaker_id": "smoke",
                "duration": 1.0,
                "feature_path": str(fp),
            }
        )
    manifest.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    dataset = FeatureDataset(manifest)
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_features)

    model = HDVRModel(
        vocab_size=64,
        text_dim=32,
        hidden_dim=32,
        identity_in_dim=8,
        identity_dim=16,
        vocal_dim=8,
        accent_feature_dim=8,
        accent_dim=16,
        prosody_feature_dim=5,
        prosody_dim=16,
        event_classes=7,
        event_dim=8,
        mel_bins=10,
        ssl_dim=12,
        channel_classes=4,
    )
    trainer = Trainer(
        model,
        build_optimizer(model.parameters(), lr=1e-3),
        CompositeLoss(),
        device="cpu",
        precision="fp32",
        output_dir=root / "checkpoint",
    )
    state = trainer.fit(
        loader,
        loader,
        max_steps=2,
        epochs=2,
        validate_every=1,
        checkpoint_every=1,
        patience=10,
        log_every=1,
    )
    checkpoint = root / "checkpoint" / "latest.pt"
    print(
        json.dumps(
            {
                "ok": checkpoint.exists(),
                "steps": state.step,
                "checkpoint": str(checkpoint),
                "manifest": str(manifest),
            },
            indent=2,
        )
    )
    raise SystemExit(0 if checkpoint.exists() else 2)


if __name__ == "__main__":
    main()
