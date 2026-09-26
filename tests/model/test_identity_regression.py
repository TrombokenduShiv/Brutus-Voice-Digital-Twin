import numpy as np

from evaluation.metrics.speaker_ecapa import identity_retention


def test_identity_retention_normalization():
    assert identity_retention(0.81,0.90) == 0.9
