from __future__ import annotations

import re


_DEVANAGARI = re.compile(r"[\u0900-\u097F]")


def detect_language(text: str) -> str:
    if _DEVANAGARI.search(text):
        return "Hindi"
    return "English"
