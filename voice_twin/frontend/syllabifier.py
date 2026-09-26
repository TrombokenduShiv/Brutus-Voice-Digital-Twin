from __future__ import annotations

_VOWEL_HINTS = set("aeiouəɪʊɛɔæɑɒʌɜɐɞœøyɨɯ")


def is_vowel_phone(phone: str) -> bool:
    p = phone.lower().replace("g:", "")
    return any(ch in _VOWEL_HINTS for ch in p)


def syllabify(phones: list[str]) -> list[list[str]]:
    if not phones:
        return []
    syllables: list[list[str]] = []
    current: list[str] = []
    seen_vowel = False
    for phone in phones:
        if is_vowel_phone(phone) and seen_vowel and current:
            syllables.append(current)
            current = []
            seen_vowel = False
        current.append(phone)
        seen_vowel |= is_vowel_phone(phone)
    if current:
        syllables.append(current)
    return syllables
