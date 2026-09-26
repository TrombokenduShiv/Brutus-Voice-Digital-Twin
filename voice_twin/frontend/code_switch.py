from __future__ import annotations

from dataclasses import dataclass

from voice_twin.frontend.language_id import detect_language


@dataclass(slots=True)
class LanguageSpan:
    text: str
    language: str


def split_code_switch(text: str) -> list[LanguageSpan]:
    words = text.split()
    if not words:
        return []
    spans: list[LanguageSpan] = []
    current_lang = detect_language(words[0])
    current: list[str] = []
    for word in words:
        lang = detect_language(word)
        if lang != current_lang and current:
            spans.append(LanguageSpan(" ".join(current), current_lang))
            current = []
            current_lang = lang
        current.append(word)
    spans.append(LanguageSpan(" ".join(current), current_lang))
    return spans
