from brier_lab.metrics import mean_log_loss, mean_forecast

def test_mean_log_loss_perfectish():
    ll = mean_log_loss([0.99, 0.01], [1, 0])
    assert ll < 0.05

def test_mean_forecast():
    assert abs(mean_forecast([0.2, 0.4, 0.6]) - 0.4) < 1e-12
