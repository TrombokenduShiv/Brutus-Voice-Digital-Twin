import numpy as np

from evaluation.metrics.accent import accent_feature_distance, allophone_accuracy


def test_identical_accent_features_have_zero_distance():
    x = np.asarray([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]])
    metrics = accent_feature_distance(x, x)
    assert metrics["accent_feature_distance"] == 0.0


def test_allophone_accuracy():
    assert allophone_accuracy([1, 2, 3], [1, 2, 3]) == 1.0
