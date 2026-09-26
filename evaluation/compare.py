from __future__ import annotations


def compare_metrics(baseline: dict[str,float], candidate: dict[str,float]) -> dict[str,float]:
    return {k: candidate[k]-baseline[k] for k in candidate.keys() & baseline.keys()}
