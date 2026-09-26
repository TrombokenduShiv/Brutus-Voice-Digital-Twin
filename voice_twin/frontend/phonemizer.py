from __future__ import annotations

import re
from dataclasses import dataclass

_WORD_RE = re.compile(r"[\w'-]+", re.UNICODE)


@dataclass(slots=True)
class PhonemeSequence:
    phones: list[str]
    word_ranges: list[tuple[str, int, int]]
    backend: str


def _fallback_word(word: str) -> list[str]:
    # Portable smoke-test fallback. Production preprocessing should use eSpeak phonemizer.
    return [f"G:{c.lower()}" for c in word if c.isalnum()]


def phonemize_text(text: str, language: str = "en-us") -> PhonemeSequence:
    words = _WORD_RE.findall(text)
    phones: list[str] = []
    ranges: list[tuple[str, int, int]] = []
    backend_name = "grapheme-fallback"
    try:
        from phonemizer.backend import EspeakBackend
        from phonemizer.separator import Separator

        lang = language.lower().replace("_", "-")
        language_map = {
            "english": "en-us",
            "hindi": "hi",
            "punjabi": "pa",
            "bengali": "bn",
            "tamil": "ta",
            "telugu": "te",
            "marathi": "mr",
        }
        lang = language_map.get(lang, lang)
        backend = EspeakBackend(
            language=lang,
            preserve_punctuation=False,
            with_stress=True,
            language_switch="remove-flags",
        )
        separator = Separator(phone="|", word=" ", syllable="")
        encoded = backend.phonemize(words, separator=separator, strip=True)
        backend_name = "espeak"
        for word, value in zip(words, encoded):
            start = len(phones)
            seq = [p for p in value.split("|") if p and not p.isspace()]
            if not seq:
                seq = _fallback_word(word)
            phones.extend(seq)
            ranges.append((word, start, len(phones)))
    except Exception:
        for word in words:
            start = len(phones)
            phones.extend(_fallback_word(word))
            ranges.append((word, start, len(phones)))
    if not phones:
        phones = ["<sil>"]
        ranges = [("", 0, 1)]
    return PhonemeSequence(phones=phones, word_ranges=ranges, backend=backend_name)


def stable_phone_id(phone: str, vocab_size: int = 2048) -> int:
    import hashlib

    if vocab_size < 16:
        raise ValueError("vocab_size must be >= 16")
    digest = hashlib.sha256(phone.encode("utf-8")).digest()
    return 2 + (int.from_bytes(digest[:8], "little") % (vocab_size - 2))
