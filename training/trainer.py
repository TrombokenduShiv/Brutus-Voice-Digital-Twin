from __future__ import annotations

import json
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import torch


@dataclass
class TrainState:
    step: int = 0
    epoch: int = 0
    best_metric: float | None = None
    bad_validations: int = 0


def move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        *,
        scheduler=None,
        device: str | torch.device = "cpu",
        precision: str = "fp32",
        accumulation_steps: int = 1,
        clip_norm: float = 1.0,
        output_dir: str | Path = "checkpoints/run",
        run_config: dict | None = None,
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.precision = precision
        self.accumulation_steps = max(1, accumulation_steps)
        self.clip_norm = clip_norm
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_config = run_config or {}
        self.state = TrainState()
        self._micro_step = 0
        self._scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.device.type == "cuda" and precision == "fp16",
        )
        self.metrics_path = self.output_dir / "metrics.jsonl"

        try:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(self.output_dir / "tensorboard")
        except Exception:
            self.writer = None

    def _autocast(self):
        if self.device.type != "cuda":
            return nullcontext()
        if self.precision == "bf16":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        if self.precision == "fp16":
            return torch.autocast("cuda", dtype=torch.float16)
        return nullcontext()

    def _write_metrics(self, split: str, metrics: dict[str, float]) -> None:
        record = {
            "split": split,
            "step": self.state.step,
            "epoch": self.state.epoch,
            **metrics,
        }
        with self.metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
        if self.writer is not None:
            for key, value in metrics.items():
                self.writer.add_scalar(f"{split}/{key}", value, self.state.step)

    def train_microbatch(self, batch: dict) -> tuple[dict[str, float], bool]:
        self.model.train()
        if self._micro_step % self.accumulation_steps == 0:
            self.optimizer.zero_grad(set_to_none=True)

        batch = move_batch(batch, self.device)
        with self._autocast():
            losses = self.model.compute_losses(batch)
            total, metrics = self.loss_fn(losses)
            scaled_total = total / self.accumulation_steps

        self._scaler.scale(scaled_total).backward()
        self._micro_step += 1
        stepped = self._micro_step % self.accumulation_steps == 0
        if stepped:
            self._scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                self.clip_norm,
            )
            self._scaler.step(self.optimizer)
            self._scaler.update()
            if self.scheduler is not None:
                self.scheduler.step()
            self.state.step += 1
        return metrics, stepped

    @torch.no_grad()
    def validate(self, loader) -> dict[str, float]:
        self.model.eval()
        sums: dict[str, float] = {}
        count = 0
        for batch in loader:
            batch = move_batch(batch, self.device)
            with self._autocast():
                losses = self.model.compute_losses(batch)
                _, metrics = self.loss_fn(losses)
            for key, value in metrics.items():
                sums[key] = sums.get(key, 0.0) + float(value)
            count += 1
        metrics = {key: value / max(1, count) for key, value in sums.items()}
        if {"speaker", "pitch", "pause"}.issubset(metrics):
            metrics["identity_prosody_composite"] = (
                metrics["speaker"] + metrics["pitch"] + metrics["pause"]
            )
        self._write_metrics("validation", metrics)
        return metrics

    def checkpoint_payload(self) -> dict:
        core_model = getattr(self.model, "student", self.model)
        return {
            "state": asdict(self.state),
            "model_state": self.model.state_dict(),
            "export_model_state": core_model.state_dict(),
            "model_config": getattr(core_model, "model_config", {}),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict() if self.scheduler else None,
            "run_config": self.run_config,
        }

    def save_checkpoint(self, name: str) -> Path:
        path = self.output_dir / name
        torch.save(self.checkpoint_payload(), path)
        return path

    def load_checkpoint(self, path: str | Path, strict: bool = True) -> None:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(payload["model_state"], strict=strict)
        if "optimizer_state" in payload:
            self.optimizer.load_state_dict(payload["optimizer_state"])
        if self.scheduler is not None and payload.get("scheduler_state"):
            self.scheduler.load_state_dict(payload["scheduler_state"])
        self.state = TrainState(**payload.get("state", {}))

    def fit(
        self,
        train_loader,
        validation_loader,
        *,
        max_steps: int,
        epochs: int = 999,
        validate_every: int = 1000,
        checkpoint_every: int = 2500,
        monitor: str = "total",
        patience: int = 15,
        log_every: int = 20,
    ) -> TrainState:
        started = perf_counter()
        for epoch in range(self.state.epoch, epochs):
            self.state.epoch = epoch
            sampler = getattr(train_loader, "batch_sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)

            for batch in train_loader:
                metrics, stepped = self.train_microbatch(batch)
                if not stepped:
                    continue

                metrics["learning_rate"] = float(self.optimizer.param_groups[0]["lr"])
                metrics["elapsed_s"] = perf_counter() - started
                if self.state.step % log_every == 0:
                    self._write_metrics("train", metrics)

                if checkpoint_every and self.state.step % checkpoint_every == 0:
                    self.save_checkpoint(f"step_{self.state.step:08d}.pt")
                    self.save_checkpoint("latest.pt")

                if (
                    validation_loader is not None
                    and validate_every
                    and self.state.step % validate_every == 0
                ):
                    val = self.validate(validation_loader)
                    metric = float(val.get(monitor, val.get("total", float("inf"))))
                    improved = (
                        self.state.best_metric is None
                        or metric < self.state.best_metric
                    )
                    if improved:
                        self.state.best_metric = metric
                        self.state.bad_validations = 0
                        self.save_checkpoint("best.pt")
                    else:
                        self.state.bad_validations += 1
                    self.save_checkpoint("latest.pt")
                    if self.state.bad_validations >= patience:
                        return self.state

                if self.state.step >= max_steps:
                    self.save_checkpoint("latest.pt")
                    return self.state

        self.save_checkpoint("latest.pt")
        return self.state
