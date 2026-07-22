from brier_lab.metrics import multi_class_brier

def test_perfect_forecast():
    assert multi_class_brier([0, 1, 0], 1) == 0.0

def test_uniform_three_class():
    s = multi_class_brier([1/3, 1/3, 1/3], 0)
    assert abs(s - (2*(1/3)**2 + (2/3)**2)) < 1e-12
