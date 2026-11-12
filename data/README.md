# Data

| File | Content | Source |
|---|---|---|
| `raw/spy_daily.csv` | SPY daily open, high, low, close and dividend-adjusted close, Jan 2003 – Nov 2026 | Yahoo Finance chart API |
| `raw/vix_daily.csv` | VIX daily open, high, low, close, Jan 1990 – Sep 2026 | Cboe, via the [datasets/finance-vix](https://github.com/datasets/finance-vix) mirror |

The study uses dates present in both files (Feb 2003 – Sep 2026 after the 22-day warm-up).
Refresh both with `python -c "from vrp import data; data.refresh()"` (needs `yfinance` and network access).
