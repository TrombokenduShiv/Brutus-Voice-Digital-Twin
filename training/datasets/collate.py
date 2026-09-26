from __future__ import annotations

import torch
from torch.nn.utils.rnn import pad_sequence


_SEQUENCE_FIELDS = (
    "token_ids",
    "accent_context",
    "prosody_context",
    "event_context",
    "acoustic_target",
    "duration_ms",
    "pause_type",
    "pause_duration_ms",
    "accent_target",
    "event_target",
)


def collate_manifest(batch: list[dict]) -> dict[str, list]:
    keys = set().union(*(item.keys() for item in batch))
    return {key: [item.get(key) for item in batch] for key in keys}


def _pad(items: list[torch.Tensor], value: float | int = 0):
    return pad_sequence(items, batch_first=True, padding_value=value)


def collate_features(batch: list[dict]) -> dict:
    lengths = torch.tensor([len(item["token_ids"]) for item in batch], dtype=torch.long)
    max_len = int(lengths.max())
    token_mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)

    out = {
        "token_ids": _pad([x["token_ids"] for x in batch], 0),
        "token_mask": token_mask,
        "identity": torch.stack([x["identity"] for x in batch]),
        "vocal": torch.stack([x["vocal"] for x in batch]),
        "speaker_memory": torch.stack([x["speaker_memory"] for x in batch]),
        "accent_context": _pad([x["accent_context"] for x in batch], 0.0),
        "prosody_context": _pad([x["prosody_context"] for x in batch], 0.0),
        "event_context": _pad([x["event_context"] for x in batch], 0.0),
        "acoustic_target": _pad([x["acoustic_target"] for x in batch], 0.0),
        "speaker_target": torch.stack([x["speaker_target"] for x in batch]),
        "ssl_target": torch.stack([x["ssl_target"] for x in batch]),
        "duration_ms": _pad([x["duration_ms"] for x in batch], 0.0),
        "pause_type": _pad([x["pause_type"] for x in batch], 0),
        "pause_duration_ms": _pad([x["pause_duration_ms"] for x in batch], 0.0),
        "accent_target": _pad([x["accent_target"] for x in batch], 0.0),
        "event_target": _pad([x["event_target"] for x in batch], 0),
        "channel_id": torch.stack([x["channel_id"] for x in batch]),
        "lengths": lengths,
        "speaker_id": [x["speaker_id"] for x in batch],
        "utterance_id": [x["utterance_id"] for x in batch],
    }
    return out
