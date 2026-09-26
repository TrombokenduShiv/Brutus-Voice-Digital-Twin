from __future__ import annotations

from training.stages.common import run_stage


def train(**kwargs):
    return run_stage("speaker-adapt", **kwargs)
