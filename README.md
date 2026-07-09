# brier-lab

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![GitHub stars](https://img.shields.io/github/stars/tapheret2/brier-lab?style=social)](https://github.com/tapheret2/brier-lab/stargazers)

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

