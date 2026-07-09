from brier_lab.metrics import accuracy_at_threshold

def test_accuracy_at_threshold():
    a = accuracy_at_threshold([1, 0, 1], [0.9, 0.1, 0.6], 0.5)
    assert abs(a - 1.0) < 1e-9
