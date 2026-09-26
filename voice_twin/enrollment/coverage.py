from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class CoverageReport:
    seconds: float
    utterances: int
    questions: int
    exclamations: int
    code_switched: int
    score: float


def coverage_report(texts: list[str], durations_s: list[float]) -> CoverageReport:
    seconds = float(sum(durations_s))
    questions = sum(t.rstrip().endswith("?") for t in texts)
    exclamations = sum(t.rstrip().endswith("!") for t in texts)
    code_switched = sum(any(ord(c) > 127 for c in t) and any("a" <= c.lower() <= "z" for c in t) for t in texts)
    diversity = min(1.0, len(texts) / 50.0)
    duration_score = min(1.0, seconds / 900.0)
    score = 0.6 * duration_score + 0.4 * diversity
    return CoverageReport(seconds, len(texts), questions, exclamations, code_switched, score)
