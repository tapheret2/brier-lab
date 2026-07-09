from brier_lab.metrics import binary_log_loss

def test_perfect_log_loss_near_zero():
    ll = binary_log_loss([1, 0], [0.999, 0.001])
    assert ll < 0.01
