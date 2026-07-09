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
