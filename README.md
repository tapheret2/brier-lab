# brier-lab

![status](https://img.shields.io/badge/status-active-brightgreen) ![python](https://img.shields.io/badge/python-3.10%2B-blue) ![license](https://img.shields.io/badge/license-MIT-lightgrey)

Minimal **forecasting metrics** toolkit (no heavy deps).

```python
from brier_lab import brier_score, log_loss, mean_brier, reliability_bins

brier_score(0.7, 1)   # (0.7-1)^2
mean_brier([0.8, 0.2], [1, 0])
```

```bash
pip install -e .
brier-lab demo
```

Useful with Polymarket / tipster ledgers / classroom forecasting.

## License

MIT
