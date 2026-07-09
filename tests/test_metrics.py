from brier_lab import brier_score, mean_brier, reliability_bins


def test_brier():
    assert brier_score(1, 1) == 0
    assert mean_brier([0.5, 0.5], [1, 0]) == 0.25


def test_bins():
    rows = reliability_bins([0.1, 0.9], [0, 1], n_bins=2)
    assert rows[0]["n"] == 1
    assert rows[1]["emp_rate"] == 1.0
