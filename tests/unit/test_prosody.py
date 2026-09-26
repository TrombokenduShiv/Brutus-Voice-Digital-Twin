import numpy as np

from evaluation.metrics.f0 import f0_metrics


def test_f0_identity():
    f=np.array([100,110,120,130],dtype=float)
    m=f0_metrics(f,f)
    assert m["f0_correlation"] > 0.999
    assert m["log_f0_rmse"] < 1e-9
