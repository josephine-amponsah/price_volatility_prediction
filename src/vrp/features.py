"""Volatility estimators, HAR inputs and the forward realised-volatility target.

Conventions
-----------
* Volatilities are annualised decimals (0.18 = 18 vol points).
* Every feature on day t uses only data known at the close of day t.
* The target on day t is realised volatility over days t+1 ... t+H (H = 21 trading days),
  the window a 30-calendar-day VIX quote prices.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
HORIZON = 21


def log_returns(adj_close: pd.Series) -> pd.Series:
    return np.log(adj_close).diff()


def parkinson_var(high: pd.Series, low: pd.Series) -> pd.Series:
    """Daily Parkinson (1980) variance from the high-low range."""
    return np.log(high / low) ** 2 / (4 * np.log(2))


def garman_klass_var(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Daily Garman-Klass (1980) variance: 0.5 ln(H/L)^2 - (2 ln 2 - 1) ln(C/O)^2."""
    return 0.5 * np.log(high / low) ** 2 - (2 * np.log(2) - 1) * np.log(close / open_) ** 2


def rolling_vol(daily_var: pd.Series, window: int) -> pd.Series:
    """Average the daily variance over the window, then annualise (never average volatilities)."""
    return np.sqrt(daily_var.rolling(window).mean() * TRADING_DAYS)


def forward_realised_vol(r: pd.Series, horizon: int = HORIZON) -> pd.Series:
    """Realised vol over t+1 ... t+horizon, aligned to day t. NaN for the last `horizon` days."""
    fwd_sum = (r ** 2).rolling(horizon).sum().shift(-horizon)
    return np.sqrt(fwd_sum * TRADING_DAYS / horizon)


def build(df: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """Feature frame indexed by date. `target` is the only forward-looking column."""
    r = log_returns(df["adj_close"])
    gk = garman_klass_var(df["open"], df["high"], df["low"], df["close"]).clip(lower=0)
    out = pd.DataFrame(index=df.index)
    out["ret"] = r
    # HAR components (Corsi, 2009) on the range-based daily variance: day, week, month
    out["har_d"] = rolling_vol(gk, 1)
    out["har_w"] = rolling_vol(gk, 5)
    out["har_m"] = rolling_vol(gk, 22)
    out["rv_cc_21"] = np.sqrt((r ** 2).rolling(horizon).mean() * TRADING_DAYS)   # close-to-close, trailing
    out["park_21"] = rolling_vol(parkinson_var(df["high"], df["low"]), 21)
    out["vix"] = df["vix"] / 100
    out["vrp_trailing"] = out["vix"] - out["rv_cc_21"]
    out["target"] = forward_realised_vol(r, horizon)
    return out.iloc[22:]
