import torch

from voice_twin.conversion.neural_converter import TwinMelConverter


def test_twin_converter_loss_graph():
    model = TwinMelConverter(
        mel_bins=16,
        hidden_dim=32,
        identity_dim=8,
        vocal_dim=8,
        style_dim=6,
        layers=2,
        heads=4,
    )
    batch = {
        "source_mel": torch.randn(2, 10, 16),
        "target_mel": torch.randn(2, 10, 16),
        "identity": torch.nn.functional.normalize(torch.randn(2, 8), dim=-1),
        "vocal": torch.randn(2, 8),
        "style": torch.randn(2, 6),
        "frame_mask": torch.ones(2, 10, dtype=torch.bool),
    }
    losses = model.compute_losses(batch)
    total = sum(losses.values())
    total.backward()
    assert all(torch.isfinite(value) for value in losses.values())
