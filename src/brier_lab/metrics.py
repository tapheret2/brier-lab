from __future__ import annotations

import math
from typing import Iterable, Sequence


def brier_score(forecast: float, outcome: int) -> float:
    f = max(0.0, min(1.0, float(forecast)))
    o = 1.0 if int(outcome) else 0.0
    return (f - o) ** 2


def log_loss(forecast: float, outcome: int, eps: float = 1e-12) -> float:
    f = max(eps, min(1 - eps, float(forecast)))
    return -math.log(f if int(outcome) else (1 - f))


def mean_brier(forecasts: Sequence[float], outcomes: Sequence[int]) -> float:
    if len(forecasts) != len(outcomes) or not forecasts:
        raise ValueError("forecasts and outcomes must be same non-empty length")
    return sum(brier_score(f, o) for f, o in zip(forecasts, outcomes)) / len(forecasts)


def reliability_bins(
    forecasts: Sequence[float],
    outcomes: Sequence[int],
    n_bins: int = 10,
) -> list[dict]:
    """Return calibration bins: avg forecast vs empirical frequency."""
    if len(forecasts) != len(outcomes) or not forecasts:
        raise ValueError("forecasts and outcomes must be same non-empty length")
    bins: list[list[tuple[float, int]]] = [[] for _ in range(n_bins)]
    for f, o in zip(forecasts, outcomes):
        f = max(0.0, min(1.0, float(f)))
        i = min(n_bins - 1, int(f * n_bins))
        bins[i].append((f, int(o)))
    out = []
    for i, items in enumerate(bins):
        if not items:
            out.append(
                {
                    "bin": i,
                    "lo": i / n_bins,
                    "hi": (i + 1) / n_bins,
                    "n": 0,
                    "avg_forecast": None,
                    "emp_rate": None,
                }
            )
            continue
        avg_f = sum(x[0] for x in items) / len(items)
        emp = sum(x[1] for x in items) / len(items)
        out.append(
            {
                "bin": i,
                "lo": i / n_bins,
                "hi": (i + 1) / n_bins,
                "n": len(items),
                "avg_forecast": avg_f,
                "emp_rate": emp,
            }
        )
    return out


def expected_calibration_error(
    forecasts: Sequence[float],
    outcomes: Sequence[int],
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (ECE) from equal-width probability bins.

    ECE = sum_i (n_i / N) * |avg_forecast_i - emp_rate_i|
    Empty bins contribute 0.
    """
    if len(forecasts) != len(outcomes) or not forecasts:
        raise ValueError("forecasts and outcomes must be same non-empty length")
    bins = reliability_bins(forecasts, outcomes, n_bins=n_bins)
    n_total = len(forecasts)
    ece = 0.0
    for row in bins:
        n = int(row["n"] or 0)
        if n == 0:
            continue
        avg_f = float(row["avg_forecast"])
        emp = float(row["emp_rate"])
        ece += (n / n_total) * abs(avg_f - emp)
    return ece


def accuracy_at_threshold(y_true, y_prob, threshold: float = 0.5) -> float:
    """Binary accuracy after thresholding probabilities."""
    if len(y_true) != len(y_prob) or not y_true:
        raise ValueError("length mismatch")
    correct = 0
    for yt, yp in zip(y_true, y_prob):
        pred = 1 if float(yp) >= threshold else 0
        if pred == int(yt):
            correct += 1
    return correct / len(y_true)


def mean_log_loss(
    forecasts: Sequence[float],
    outcomes: Sequence[int],
    eps: float = 1e-12,
) -> float:
    """Mean binary log-loss over paired forecasts/outcomes."""
    if len(forecasts) != len(outcomes) or not forecasts:
        raise ValueError("forecasts and outcomes length mismatch")
    return sum(log_loss(f, o, eps=eps) for f, o in zip(forecasts, outcomes)) / len(
        forecasts
    )


def mean_forecast(forecasts: Sequence[float]) -> float:
    """Average predicted probability (sharpness / base-rate check)."""
    if not forecasts:
        raise ValueError("forecasts must be non-empty")
    return sum(float(f) for f in forecasts) / len(forecasts)


def sharpness(forecasts: Sequence[float]) -> float:
    """Variance of forecasts around their mean (higher = sharper)."""
    if not forecasts:
        raise ValueError("forecasts must be non-empty")
    mu = sum(float(f) for f in forecasts) / len(forecasts)
    return sum((float(f) - mu) ** 2 for f in forecasts) / len(forecasts)


def brier_skill_score(
    forecasts: Sequence[float],
    outcomes: Sequence[int],
    ref: float | None = None,
) -> float:
    """1 - BS/BS_ref where ref is climatology (mean outcome) by default."""
    if len(forecasts) != len(outcomes) or not forecasts:
        raise ValueError("length mismatch")
    bs = mean_brier(forecasts, outcomes)
    if ref is None:
        ref = sum(int(o) for o in outcomes) / len(outcomes)
    ref = max(0.0, min(1.0, float(ref)))
    bs_ref = mean_brier([ref] * len(outcomes), outcomes)
    if bs_ref == 0:
        return 0.0 if bs == 0 else float("-inf")
    return 1.0 - (bs / bs_ref)


def multi_class_brier(probs: Sequence[float], outcome_index: int) -> float:
    """Multi-class Brier score for one forecast distribution.

    ``probs`` should sum ~1; ``outcome_index`` is the true class index.
    """
    if not probs:
        raise ValueError("probs must be non-empty")
    if not 0 <= outcome_index < len(probs):
        raise ValueError("outcome_index out of range")
    total = 0.0
    for i, p in enumerate(probs):
        y = 1.0 if i == outcome_index else 0.0
        total += (float(p) - y) ** 2
    return total


def log_loss_binary(y_true: float, p: float, eps: float = 1e-15) -> float:
    """Binary log loss for a single outcome (0/1) and probability p."""
    if y_true not in (0, 1, 0.0, 1.0):
        raise ValueError("y_true must be 0 or 1")
    p = min(1.0 - eps, max(eps, float(p)))
    import math
    if y_true == 1:
        return -math.log(p)
    return -math.log(1.0 - p)


def mean_absolute_error(y_true: list[float], y_pred: list[float]) -> float:
    if len(y_true) != len(y_pred) or not y_true:
        raise ValueError("non-empty equal-length lists required")
    return sum(abs(a - b) for a, b in zip(y_true, y_pred)) / len(y_true)
