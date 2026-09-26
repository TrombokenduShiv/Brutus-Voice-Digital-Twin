from __future__ import annotations

import torch

from training.losses.total import CompositeLoss
from training.optimizer import build_optimizer
from training.trainer import Trainer
from voice_twin.hdvr.model import HDVRModel


def _batch(batch_size=2, tokens=5, memory_slots=4):
    mask = torch.ones(batch_size, tokens, dtype=torch.bool)
    return {
        "token_ids": torch.randint(2, 32, (batch_size, tokens)),
        "token_mask": mask,
        "identity": torch.randn(batch_size, 8),
        "vocal": torch.randn(batch_size, 8),
        "speaker_memory": torch.randn(batch_size, memory_slots, 8),
        "accent_context": torch.randn(batch_size, tokens, 8),
        "prosody_context": torch.rand(batch_size, tokens, 5),
        "prosody_target": torch.cat(
            [
                torch.rand(batch_size, tokens, 1) * 200 + 80,
                torch.rand(batch_size, tokens, 4),
            ],
            dim=-1,
        ),
        "event_context": torch.nn.functional.one_hot(
            torch.zeros(batch_size, tokens, dtype=torch.long),
            num_classes=7,
        ).float(),
        "acoustic_target": torch.rand(batch_size, tokens, 10),
        "speaker_target": torch.nn.functional.normalize(
            torch.randn(batch_size, 8), dim=-1
        ),
        "ssl_target": torch.randn(batch_size, 12),
        "duration_ms": torch.rand(batch_size, tokens) * 150 + 20,
        "pause_type": torch.zeros(batch_size, tokens, dtype=torch.long),
        "pause_duration_ms": torch.rand(batch_size, tokens) * 50,
        "accent_target": torch.rand(batch_size, tokens, 8),
        "event_target": torch.zeros(batch_size, tokens, dtype=torch.long),
        "channel_id": torch.zeros(batch_size, dtype=torch.long),
    }


def test_hdvr_full_loss_graph_and_checkpoint(tmp_path):
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
    optimizer = build_optimizer(model.parameters(), lr=1e-3)
    trainer = Trainer(
        model,
        optimizer,
        CompositeLoss(),
        output_dir=tmp_path,
        precision="fp32",
    )
    metrics, stepped = trainer.train_microbatch(_batch())
    assert stepped is True
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    path = trainer.save_checkpoint("smoke.pt")
    assert path.exists()
