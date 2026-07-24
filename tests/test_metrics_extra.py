import math
from brier_lab.metrics import log_loss_binary, mean_absolute_error


def test_log_loss_binary():
    assert abs(log_loss_binary(1, 1.0) - 0.0) < 1e-9
    assert log_loss_binary(1, 0.5) == math.log(2)


def test_mean_absolute_error():
    assert mean_absolute_error([1, 0], [0.5, 0.5]) == 0.5
