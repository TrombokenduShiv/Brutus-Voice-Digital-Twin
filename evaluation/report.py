from __future__ import annotations

import json
from pathlib import Path


def write_report(metrics: dict, output: str | Path) -> Path:
    path=Path(output); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(metrics,indent=2,sort_keys=True),encoding="utf-8")
    return path
