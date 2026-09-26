from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch import nn

from training.losses.total import CompositeLoss
from training.optimizer import build_optimizer
from training.scheduler import cosine_with_warmup
from training.stages.common import build_loader, build_model, load_yaml, resolve_device
from training.trainer import Trainer


class StreamingDistillationModel(nn.Module):
    def __init__(self, teacher):
        super().__init__()
        self.teacher = teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.student = copy.deepcopy(teacher)
        self.model_config = dict(getattr(teacher, "model_config", {}))

    def compute_losses(self, batch):
        with torch.no_grad():
            target = self.teacher(batch, causal=False)
        pred = self.student(batch, causal=True)
        mask = batch["token_mask"].unsqueeze(-1).to(pred["conditioning"].dtype)
        denom = mask.sum().clamp_min(1.0)
        acoustic_distillation = (
            ((pred["conditioning"] - target["conditioning"]) ** 2) * mask
        ).sum() / denom
        speaker_consistency = (
            1.0
            - F.cosine_similarity(
                pred["speaker_embedding"],
                target["speaker_embedding"],
                dim=-1,
            )
        ).mean()
        prosody_continuity = (
            (pred["prosody"] - target["prosody"]).abs() * mask
        ).sum() / denom
        return {
            "acoustic_distillation": acoustic_distillation,
            "speaker_consistency": speaker_consistency,
            "prosody_continuity": prosody_continuity,
        }


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
    if not base_checkpoint:
        raise ValueError("streaming-distill requires --base-checkpoint")
    config = load_yaml(config_path)
    section = config.get("training", config)
    teacher = build_model(model_config_path, base_checkpoint)
    wrapper = StreamingDistillationModel(teacher)
    params = [p for p in wrapper.student.parameters() if p.requires_grad]
    optimizer = build_optimizer(
        params,
        lr=float(section.get("learning_rate", 1e-4)),
        weight_decay=float(section.get("weight_decay", 0.01)),
    )
    max_steps = int(section.get("max_steps", 40000))
    scheduler = cosine_with_warmup(
        optimizer,
        int(section.get("warmup_steps", min(2000, max_steps // 20))),
        max_steps,
    )
    weights = dict(section.get("losses", {}))
    trainer = Trainer(
        wrapper,
        optimizer,
        CompositeLoss(weights),
        scheduler=scheduler,
        device=resolve_device(device),
        precision=str(section.get("precision", "bf16")),
        accumulation_steps=int(section.get("accumulation_steps", 1)),
        clip_norm=float(section.get("clip_norm", 1.0)),
        output_dir=output_dir,
        run_config={
            "stage": "streaming-distill",
            "config": config,
            "base_checkpoint": base_checkpoint,
        },
    )
    if resume:
        trainer.load_checkpoint(resume, strict=False)
    train_loader = build_loader(train_manifest, section, shuffle=True)
    val_loader = (
        build_loader(validation_manifest, section, shuffle=False)
        if validation_manifest
        else None
    )
    return trainer.fit(
        train_loader,
        val_loader,
        max_steps=max_steps,
        epochs=int(section.get("epochs", 999)),
        validate_every=int(section.get("validate_every_steps", 500)),
        checkpoint_every=int(section.get("checkpoint_every_steps", 1000)),
        monitor="total",
        patience=int(section.get("patience_evaluations", 10)),
    )
