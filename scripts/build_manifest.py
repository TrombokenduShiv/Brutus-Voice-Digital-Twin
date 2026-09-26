from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import soundfile as sf


REQUIRED = {"audio", "text", "speaker_id"}


def _read_metadata(path: Path) -> list[dict]:
    if path.suffix.lower() == ".jsonl":
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _stable_key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _assign_splits(rows: list[dict]) -> None:
    pending = [r for r in rows if not r.get("split")]
    groups: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in pending:
        speaker = str(row["speaker_id"])
        session = str(
            row.get("session_id")
            or row.get("recording_day")
            or row.get("session")
            or row.get("id")
            or row["audio"]
        )
        groups[speaker][session].append(row)

    for speaker, sessions in groups.items():
        keys = sorted(sessions, key=lambda s: _stable_key(f"{speaker}|{s}"))
        n = len(keys)
        if n < 3:
            allocation = {key: "train" for key in keys}
        else:
            n_test = max(1, round(n * 0.15))
            n_val = max(1, round(n * 0.15))
            n_train = max(1, n - n_val - n_test)
            allocation = {}
            for i, key in enumerate(keys):
                if i < n_train:
                    allocation[key] = "train"
                elif i < n_train + n_val:
                    allocation[key] = "validation"
                else:
                    allocation[key] = "test"
        for session, session_rows in sessions.items():
            for row in session_rows:
                row["split"] = allocation[session]


def build_manifest(metadata: Path, output_dir: Path, audio_root: Path | None = None) -> dict[str, int]:
    rows = _read_metadata(metadata)
    if not rows:
        raise ValueError("metadata contains no utterances")
    missing = REQUIRED.difference(rows[0])
    if missing:
        raise ValueError(f"metadata missing required columns: {sorted(missing)}")

    root = audio_root or metadata.parent
    cleaned: list[dict] = []
    for index, raw in enumerate(rows):
        row = {k: v for k, v in raw.items() if v not in (None, "")}
        audio = Path(str(row["audio"]))
        if not audio.is_absolute():
            audio = (root / audio).resolve()
        if not audio.exists():
            raise FileNotFoundError(audio)
        row["audio"] = str(audio)
        row["id"] = str(row.get("id") or f"{row['speaker_id']}_{index:07d}")
        row["language"] = str(row.get("language", "en-us"))
        row["session_id"] = str(
            row.get("session_id")
            or row.get("recording_day")
            or row.get("session")
            or "session_0"
        )
        if "duration" not in row:
            row["duration"] = float(sf.info(str(audio)).duration)
        else:
            row["duration"] = float(row["duration"])
        cleaned.append(row)

    _assign_splits(cleaned)
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = {"train": 0, "validation": 0, "test": 0}
    handles = {
        split: (output_dir / f"{split}.jsonl").open("w", encoding="utf-8")
        for split in counts
    }
    try:
        for row in cleaned:
            split = str(row.get("split", "train")).lower()
            if split == "val":
                split = "validation"
            if split not in handles:
                raise ValueError(f"unsupported split '{split}' for {row['id']}")
            handles[split].write(json.dumps(row, ensure_ascii=False) + "\n")
            counts[split] += 1
    finally:
        for handle in handles.values():
            handle.close()
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Build leakage-resistant BRUTUS JSONL manifests.")
    parser.add_argument("--metadata", required=True, help="CSV or JSONL metadata.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--audio-root")
    args = parser.parse_args()
    counts = build_manifest(
        Path(args.metadata),
        Path(args.output_dir),
        Path(args.audio_root) if args.audio_root else None,
    )
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
