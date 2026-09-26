from __future__ import annotations

import re


_WS = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    text = text.strip()
    text = text.replace("…", "...")
    return _WS.sub(" ", text)
