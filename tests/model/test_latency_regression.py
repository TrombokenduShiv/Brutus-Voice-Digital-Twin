from evaluation.metrics.latency import realtime_factor


def test_realtime_gate_example():
    assert realtime_factor(0.2,1.0) < 0.5
