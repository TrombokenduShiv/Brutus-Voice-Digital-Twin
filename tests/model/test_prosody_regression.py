import numpy as np

from evaluation.metrics.f0 import f0_metrics
from evaluation.metrics.pauses import pause_metrics
from voice_twin.prosody.pause_detector import Pause


def test_identical_f0_is_perfect():
    f0 = np.asarray([100.0, 110.0, 120.0, 130.0])
    metrics = f0_metrics(f0, f0)
    assert metrics["f0_correlation"] > 0.999
    assert metrics["log_f0_rmse"] < 1e-9


def test_identical_pauses_are_perfect():
    pauses = [Pause(1.0, 1.2, 0.2)]
    metrics = pause_metrics(pauses, pauses)
    assert metrics["pause_f1"] == 1.0
    assert metrics["pause_start_mae_ms"] == 0.0
    assert metrics["pause_duration_mae_ms"] == 0.0
