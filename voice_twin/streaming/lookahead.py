from __future__ import annotations


def split_phrases(text: str, max_chars: int = 180) -> list[str]:
    phrases, current = [], ""
    for token in text.split():
        candidate = f"{current} {token}".strip()
        boundary = token.endswith((".", "?", "!", ";", ":"))
        if current and (len(candidate) > max_chars or boundary):
            current = candidate
            phrases.append(current)
            current = ""
        else:
            current = candidate
    if current:
        phrases.append(current)
    return phrases
