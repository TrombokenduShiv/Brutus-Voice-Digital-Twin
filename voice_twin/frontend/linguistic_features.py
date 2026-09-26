from __future__ import annotations

from dataclasses import dataclass

from voice_twin.frontend.text_normalizer import normalize_text


@dataclass(slots=True)
class LinguisticFeatures:
    text: str
    tokens: list[str]
    punctuation: list[str]
    is_question: bool
    word_count: int


def extract_linguistic_features(text: str) -> LinguisticFeatures:
    t = normalize_text(text)
    tokens = t.split()
    punctuation = [c for c in t if c in ",.;:?!"]
    return LinguisticFeatures(t, tokens, punctuation, t.endswith("?"), len(tokens))
