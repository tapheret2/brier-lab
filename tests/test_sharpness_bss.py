from brier_lab.metrics import brier_skill_score, sharpness

def test_sharpness_constant_is_zero():
    assert sharpness([0.4, 0.4, 0.4]) < 1e-12

def test_bss_perfect_better_than_climatology():
    # perfect forecasts vs outcomes
    f = [1.0, 0.0, 1.0, 0.0]
    o = [1, 0, 1, 0]
    assert brier_skill_score(f, o) > 0.9
