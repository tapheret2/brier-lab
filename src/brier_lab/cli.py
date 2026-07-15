from __future__ import annotations

from brier_lab.metrics import (
    brier_score,
    expected_calibration_error,
    mean_brier,
    reliability_bins,
)


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(prog="brier-lab")
    p.add_argument("cmd", nargs="?", default="demo", choices=["demo", "ece"])
    args = p.parse_args()
    if args.cmd in {"demo", "ece"}:
        f = [0.9, 0.8, 0.2, 0.1, 0.55]
        o = [1, 1, 0, 0, 1]
        print("mean_brier", mean_brier(f, o))
        print("ece", expected_calibration_error(f, o, n_bins=5))
        print("single", brier_score(0.7, 1))
        for row in reliability_bins(f, o, n_bins=5):
            if row["n"]:
                print(row)


if __name__ == "__main__":
    main()
