from __future__ import annotations

from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from training.datasets.collate import collate_features
from training.datasets.dataset import FeatureDataset
from training.datasets.sampler import SecondsBatchSampler
from training.losses.total import CompositeLoss, DEFAULT_WEIGHTS
from training.optimizer import build_optimizer
from training.scheduler import cosine_with_warmup
from training.trainer import Trainer
from voice_twin.hdvr.model import HDVRModel


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def model_kwargs(config: dict) -> dict:
    h = config["hdvr"]
    return {
        "vocab_size": h.get("vocab_size", 2048),
        "text_dim": h.get("text_dim", 512),
        "hidden_dim": h.get("hidden_dim", 512),
        "identity_in_dim": h["identity"].get("input_dim", 192),
        "identity_dim": h["identity"].get("projected_dim", 256),
        "vocal_dim": h["vocal_profile"].get("dim", 128),
        "accent_feature_dim": h["accent"].get("feature_dim", 8),
        "accent_dim": h["accent"].get("latent_dim", 192),
        "prosody_feature_dim": h["prosody"].get("feature_dim", 5),
        "prosody_dim": h["prosody"].get("latent_dim", 256),
        "event_classes": len(h["events"].get("classes", [])) or 7,
        "event_dim": h["events"].get("latent_dim", 64),
        "mel_bins": h["acoustic"].get("mel_bins", 80),
        "ssl_dim": h["ssl"].get("dim", 768),
        "channel_classes": h["channel"].get("classes", 32),
        "channel_grl_weight": h["channel"].get("gradient_reversal_weight", 1.0),
    }


def build_model(model_config_path: str | Path, checkpoint: str | Path | None = None) -> HDVRModel:
    if checkpoint:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        kwargs = payload.get("model_config") or model_kwargs(load_yaml(model_config_path))
        model = HDVRModel(**kwargs)
        state = payload.get("export_model_state") or payload.get("model_state")
        model.load_state_dict(state, strict=False)
        return model
    return HDVRModel(**model_kwargs(load_yaml(model_config_path)))


def build_loader(
    manifest: str | Path,
    section: dict,
    *,
    shuffle: bool,
) -> DataLoader:
    dataset = FeatureDataset(manifest)
    batch_cfg = section.get("batch", {})
    max_seconds = float(batch_cfg.get("max_audio_seconds_per_gpu", 0) or 0)
    batch_size = int(section.get("batch_size", batch_cfg.get("max_items", 16)))
    if max_seconds > 0:
        sampler = SecondsBatchSampler(
            dataset,
            max_seconds=max_seconds,
            max_items=batch_size,
            shuffle=shuffle,
        )
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=collate_features,
            num_workers=int(section.get("num_workers", 2)),
            pin_memory=True,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_features,
        num_workers=int(section.get("num_workers", 2)),
        pin_memory=True,
    )


def stage_weights(stage: str, section: dict) -> dict[str, float]:
    if stage == "hdvr":
        return dict(section.get("loss_weights", DEFAULT_WEIGHTS))
    if stage == "accent":
        losses = section.get("losses", {})
        return {
            "accent": float(losses.get("allophone", 1.0)),
            "duration": float(losses.get("duration", 0.3)),
            "acoustic": float(losses.get("acoustic_feature", 0.3)),
        }
    if stage == "prosody":
        losses = section.get("losses", {})
        return {
            "pitch": float(losses.get("log_f0", 1.0)),
            "duration": float(losses.get("duration", 0.5)),
            "pause": float(losses.get("pause", 0.8)),
            "event": float(losses.get("breath_probability", 0.5)),
        }
    if stage == "events":
        return {"event": 1.0}
    if stage == "speaker-adapt":
        return {
            "acoustic": 0.50,
            "speaker": 1.00,
            "pitch": 0.50,
            "duration": 0.25,
            "pause": 0.50,
            "accent": 0.50,
            "event": 0.20,
            "ssl_perceptual": 0.50,
        }
    raise ValueError(f"unknown stage: {stage}")


def configure_stage(model: HDVRModel, stage: str, section: dict) -> None:
    if stage == "hdvr":
        return
    if stage == "accent":
        model.freeze_except(("accent_encoder", "accent_head", "duration_head", "acoustic_head"))
    elif stage == "prosody":
        model.freeze_except(
            (
                "prosody_encoder",
                "prosody_head",
                "duration_head",
                "pause_type_head",
                "pause_duration_head",
                "event_head",
            )
        )
    elif stage == "events":
        model.freeze_except(("event_encoder", "event_head"))
    elif stage == "speaker-adapt":
        lora = section.get("lora", {})
        model.enable_speaker_adapter(
            rank=int(lora.get("rank", 16)),
            alpha=int(lora.get("alpha", 32)),
            dropout=float(lora.get("dropout", 0.05)),
        )
        model.freeze_for_speaker_adaptation()
    else:
        raise ValueError(stage)


def run_stage(
    stage: str,
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
    config = load_yaml(config_path)
    section = config.get("training") or config.get("adaptation") or config
    resolved_device = resolve_device(device)

    if stage == "speaker-adapt" and not base_checkpoint:
        raise ValueError("speaker-adapt requires --base-checkpoint")

    model = build_model(model_config_path, base_checkpoint)
    configure_stage(model, stage, section)
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise RuntimeError(f"stage {stage} has no trainable parameters")

    optimizer_cfg = section.get("optimizer", {})
    optimizer = build_optimizer(
        parameters,
        lr=float(optimizer_cfg.get("learning_rate", 2e-4)),
        weight_decay=float(optimizer_cfg.get("weight_decay", 0.01)),
    )
    max_steps = int(section.get("max_steps", 250000))
    scheduler_cfg = section.get("scheduler", {})
    scheduler = cosine_with_warmup(
        optimizer,
        int(scheduler_cfg.get("warmup_steps", min(5000, max_steps // 20))),
        max_steps,
    )

    train_loader = build_loader(train_manifest, section, shuffle=True)
    val_loader = (
        build_loader(validation_manifest, section, shuffle=False)
        if validation_manifest
        else None
    )
    gradient = section.get("gradient", {})
    early = section.get("early_stopping", {})
    trainer = Trainer(
        model,
        optimizer,
        CompositeLoss(stage_weights(stage, section)),
        scheduler=scheduler,
        device=resolved_device,
        precision=str(section.get("precision", "bf16")),
        accumulation_steps=int(gradient.get("accumulation_steps", 1)),
        clip_norm=float(
            gradient.get("clip_norm", section.get("gradient_clip_norm", 1.0))
        ),
        output_dir=output_dir,
        run_config={
            "stage": stage,
            "config": config,
            "model_config": load_yaml(model_config_path),
            "train_manifest": train_manifest,
            "validation_manifest": validation_manifest,
            "base_checkpoint": base_checkpoint,
        },
    )
    if resume:
        trainer.load_checkpoint(resume, strict=False)

    monitor = str(early.get("metric", "total"))
    return trainer.fit(
        train_loader,
        val_loader,
        max_steps=max_steps,
        epochs=int(section.get("epochs", 999)),
        validate_every=int(section.get("validate_every_steps", 1000)),
        checkpoint_every=int(section.get("checkpoint_every_steps", 2500)),
        monitor=monitor,
        patience=int(early.get("patience_evaluations", 15)),
        log_every=int(section.get("log_every_steps", 20)),
    )
