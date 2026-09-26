from __future__ import annotations

import torch
import yaml
from torch.utils.data import DataLoader

from training.conversion_dataset import ConversionDataset, collate_conversion
from training.losses.total import CompositeLoss
from training.optimizer import build_optimizer
from training.scheduler import cosine_with_warmup
from training.stages.common import resolve_device
from training.trainer import Trainer
from voice_twin.conversion.neural_converter import TwinMelConverter


def _load_model(checkpoint: str | None) -> TwinMelConverter:
    if not checkpoint:
        return TwinMelConverter()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = payload.get("model_config", {})
    model = TwinMelConverter(**config)
    state = payload.get("export_model_state") or payload["model_state"]
    model.load_state_dict(state, strict=False)
    return model


def train(
    *,
    config_path: str,
    model_config_path: str,
    train_manifest: str,
    validation_manifest: str | None,
    output_dir: str,
    device: str = "auto",
    base_checkpoint: str | None = None,
    resume: str | None = None,
):
    del model_config_path
    with open(config_path, "r", encoding="utf-8") as f:
        section = (yaml.safe_load(f) or {}).get("training", {})
    model = _load_model(base_checkpoint)
    optimizer = build_optimizer(
        model.parameters(),
        lr=float(section.get("learning_rate", 2e-4)),
        weight_decay=float(section.get("weight_decay", 0.01)),
    )
    max_steps = int(section.get("max_steps", 100000))
    scheduler = cosine_with_warmup(
        optimizer,
        int(section.get("warmup_steps", 3000)),
        max_steps,
    )

    def loader(path, shuffle):
        ds = ConversionDataset(path)
        return DataLoader(
            ds,
            batch_size=int(section.get("batch_size", 8)),
            shuffle=shuffle,
            num_workers=int(section.get("num_workers", 2)),
            pin_memory=True,
            collate_fn=collate_conversion,
        )

    weights = section.get(
        "losses",
        {
            "conversion_mel": 1.0,
            "conversion_delta": 0.5,
            "conversion_identity": 0.3,
        },
    )
    trainer = Trainer(
        model,
        optimizer,
        CompositeLoss(weights),
        scheduler=scheduler,
        device=resolve_device(device),
        precision=str(section.get("precision", "bf16")),
        accumulation_steps=int(section.get("accumulation_steps", 2)),
        clip_norm=float(section.get("clip_norm", 1.0)),
        output_dir=output_dir,
        run_config={
            "stage": "twin-converter",
            "config": section,
            "base_checkpoint": base_checkpoint,
        },
    )
    if resume:
        trainer.load_checkpoint(resume, strict=False)
    return trainer.fit(
        loader(train_manifest, True),
        loader(validation_manifest, False) if validation_manifest else None,
        max_steps=max_steps,
        epochs=int(section.get("epochs", 999)),
        validate_every=int(section.get("validate_every_steps", 500)),
        checkpoint_every=int(section.get("checkpoint_every_steps", 1000)),
        monitor="total",
        patience=int(section.get("patience_evaluations", 12)),
    )
