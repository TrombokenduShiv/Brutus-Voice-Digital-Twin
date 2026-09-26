from __future__ import annotations


def collate_manifest(batch: list[dict]) -> dict[str, list]:
    keys = set().union(*(item.keys() for item in batch))
    return {key: [item.get(key) for item in batch] for key in keys}
