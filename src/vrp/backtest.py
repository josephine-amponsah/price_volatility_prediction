"""Volatility risk premium strategy: a rolling one-month variance swap on SPY.

Position
--------
Every 21 trading days the strategy decides whether to sell a 21-day variance swap struck at the
current VIX. Short variance earns  (K^2 - RV^2) / (2K)  per unit of vega notional, where K and RV
are in vol points: it wins when realised volatility comes in below the strike.

The P&L is accrued daily on the realised leg: day s contributes
    (K^2 / 21 - 252 * 100^2 * r_s^2 / 21) / (2K),
which sums exactly to the swap's payoff at expiry. Mark-to-market moves in implied volatility
before expiry are not modelled (see README, Limitations).

Units: P&L is in vol points per 1 unit of vega notional.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .features import HORIZON, TRADING_DAYS
from .models import OOS_START


def roll_dates(index: pd.DatetimeIndex, start: str = OOS_START, every: int = HORIZON) -> pd.DatetimeIndex:
    first = int(index.searchsorted(pd.Timestamp(start)))
    return index[first::every]


def variance_swap_pnl(feat: pd.DataFrame, signal: pd.Series, cost_vol_pts: float = 0.5,
                      start: str = OOS_START, every: int = HORIZON) -> pd.DataFrame:
    """Daily P&L of a short-variance strategy.

    `signal` is evaluated on roll dates only: -1 = sell variance, 0 = flat (1 = buy, allowed).
    The trade is struck at the roll-date close, so day t's signal earns from day t+1.
    `cost_vol_pts` is charged in vol points of vega each time a swap is opened.
    """
    rolls = roll_dates(feat.index, start, every)
    k_vol = (feat["vix"] * 100).reindex(rolls)
    pos_at_roll = signal.reindex(rolls).fillna(0)

    pos = pd.Series(np.nan, index=feat.index)
    strike = pd.Series(np.nan, index=feat.index)
    pos.loc[rolls] = pos_at_roll.values
    strike.loc[rolls] = k_vol.values
    # carry the roll-date decision through the next 21 days, starting the day after the roll
    pos = pos.ffill().shift(1)
    strike = strike.ffill().shift(1)

    realised_var_pts = TRADING_DAYS * (feat["ret"] * 100) ** 2               # annualised, vol pts^2
    # long variance (pos = +1) earns (RV^2 - K^2) / (2K); short (pos = -1) earns the opposite
    daily = pos * (realised_var_pts - strike ** 2) / (2 * strike) / every
    opened = pd.Series(0.0, index=feat.index)
    opened.loc[rolls] = pos_at_roll.abs().values
    costs = opened.shift(1).fillna(0) * cost_vol_pts
    out = pd.DataFrame({"position": pos, "strike": strike, "gross": daily, "cost": costs})
    out["pnl"] = out["gross"] - out["cost"]
    return out.loc[out["position"].notna()]


def summary(pnl: pd.Series, periods_per_year: int = TRADING_DAYS) -> dict:
    """Sharpe, drawdown, hit rate and tail statistics of a daily P&L series (vol points)."""
    pnl = pnl.dropna()
    equity = pnl.cumsum()
    monthly = pnl.groupby(np.arange(len(pnl)) // HORIZON).sum()
    active = monthly[monthly != 0]
    return {
        "total_pnl": float(equity.iloc[-1]),
        "ann_return": float(pnl.mean() * periods_per_year),
        "ann_vol": float(pnl.std() * np.sqrt(periods_per_year)),
        "sharpe": float(pnl.mean() / pnl.std() * np.sqrt(periods_per_year)),
        "max_drawdown": float((equity - equity.cummax()).min()),
        "worst_month": float(monthly.min()),
        "hit_rate_months": float((active > 0).mean()) if len(active) else np.nan,
        "share_months_invested": float((monthly != 0).mean()),
        "skew_monthly": float(stats.skew(monthly)),
    }


def deflated_sharpe(returns: pd.Series, trial_sharpes: list[float]) -> dict:
    """Deflated Sharpe ratio (Bailey and Lopez de Prado, 2014) on per-period returns.

    `trial_sharpes` are the per-period Sharpe ratios of every variant tried, including this one.
    Returns the probability that the true Sharpe exceeds the best one expected from luck alone.
    """
    r = returns.dropna()
    n_obs, n_trials = len(r), len(trial_sharpes)
    sr = r.mean() / r.std()
    var_sr = np.var(trial_sharpes, ddof=1) if n_trials > 1 else 0.0
    gamma = 0.5772156649
    z = stats.norm.ppf
    sr0 = np.sqrt(var_sr) * ((1 - gamma) * z(1 - 1 / n_trials) + gamma * z(1 - 1 / (n_trials * np.e))) \
        if n_trials > 1 else 0.0
    skew, kurt = stats.skew(r), stats.kurtosis(r, fisher=False)
    denom = np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr ** 2)
    dsr = stats.norm.cdf((sr - sr0) * np.sqrt(n_obs - 1) / denom)
    return {"sharpe_per_period": float(sr), "expected_max_sharpe_from_luck": float(sr0),
            "deflated_sharpe": float(dsr), "n_trials": n_trials, "n_obs": n_obs}
