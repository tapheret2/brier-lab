from brier_lab.metrics import expected_calibration_error


def test_perfect_calibration_low_ece():
    # Perfectly calibrated extremes
    f = [0.0, 0.0, 1.0, 1.0]
    o = [0, 0, 1, 1]
    ece = expected_calibration_error(f, o, n_bins=4)
    assert ece == 0.0


def test_miscalibrated_positive_ece():
    f = [0.9, 0.9, 0.9, 0.9]
    o = [0, 0, 0, 1]
    ece = expected_calibration_error(f, o, n_bins=5)
    assert ece > 0.0
