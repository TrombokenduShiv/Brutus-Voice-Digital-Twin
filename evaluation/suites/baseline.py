from __future__ import annotations

from evaluation.suites.cloning import evaluate_clone


def evaluate_baseline(*args, **kwargs):
    return evaluate_clone(*args, **kwargs)
