"""Market risk: one-day Value at Risk and Expected Shortfall, and their backtests.

Two VaR models are compared on daily SPY returns:
  * GJR-GARCH(1,1) with Student-t errors (conditional, reacts to volatility clustering)
  * 250-day historical simulation (unconditional, the common desk baseline)

VaR and ES are reported as positive loss numbers (decimal returns).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def student_t_var_es(mu: pd.Series, sigma: pd.Series, nu: pd.Series, var_level: float = 0.99,
                     es_level: float = 0.975) -> pd.DataFrame:
    """VaR and ES of a location-scale Student-t with unit variance (as in GARCH-t)."""
    a_var, a_es = 1 - var_level, 1 - es_level
    scale = np.sqrt((nu - 2) / nu)                     # standardised t has variance 1
    q_var = stats.t.ppf(a_var, nu) * scale
    q_es = stats.t.ppf(a_es, nu)
    # E[T | T < q] for a standard t, then rescaled to unit variance
    tail_mean = -stats.t.pdf(q_es, nu) / a_es * (nu + q_es ** 2) / (nu - 1) * scale
    return pd.DataFrame({"var": -(mu + sigma * q_var), "es": -(mu + sigma * tail_mean)}, index=mu.index)


def historical_var_es(returns: pd.Series, window: int = 250, var_level: float = 0.99,
                      es_level: float = 0.975) -> pd.DataFrame:
    """Rolling historical-simulation VaR and ES, using only the previous `window` days."""
    past = returns.shift(1).rolling(window)
    var = -past.quantile(1 - var_level)
    es = -returns.shift(1).rolling(window).apply(
        lambda x: x[x <= np.quantile(x, 1 - es_level)].mean(), raw=True)
    return pd.DataFrame({"var": var, "es": es})


def kupiec(breaches: pd.Series, level: float = 0.99) -> dict:
    """Proportion-of-failures test (Kupiec, 1995). H0: breach rate = 1 - level."""
    b = breaches.dropna().astype(bool)
    n, x, p = len(b), int(b.sum()), 1 - level
    phat = x / n
    ll0 = (n - x) * np.log(1 - p) + x * np.log(p)
    ll1 = ((n - x) * np.log(1 - phat) if phat < 1 else 0.0) + (x * np.log(phat) if x > 0 else 0.0)
    lr = -2 * (ll0 - ll1)
    return {"n": n, "breaches": x, "expected": n * p, "breach_rate": phat,
            "lr": float(lr), "p_value": float(1 - stats.chi2.cdf(lr, 1))}


def christoffersen(breaches: pd.Series, level: float = 0.99) -> dict:
    """Independence test (Christoffersen, 1998): are breaches clustered in time?"""
    b = breaches.dropna().astype(int).to_numpy()
    prev, curr = b[:-1], b[1:]
    n00 = int(((prev == 0) & (curr == 0)).sum())
    n01 = int(((prev == 0) & (curr == 1)).sum())
    n10 = int(((prev == 1) & (curr == 0)).sum())
    n11 = int(((prev == 1) & (curr == 1)).sum())
    pi0 = n01 / max(n00 + n01, 1)
    pi1 = n11 / max(n10 + n11, 1)
    pi = (n01 + n11) / max(len(curr), 1)

    def ll(p, a, b_):
        return (a * np.log(1 - p) if a and p < 1 else 0.0) + (b_ * np.log(p) if b_ and p > 0 else 0.0)

    lr_ind = -2 * (ll(pi, n00 + n10, n01 + n11) - ll(pi0, n00, n01) - ll(pi1, n10, n11))
    lr_cc = lr_ind + kupiec(breaches, level)["lr"]
    return {"p_after_breach": pi1, "p_after_no_breach": pi0,
            "p_value_independence": float(1 - stats.chi2.cdf(lr_ind, 1)),
            "p_value_conditional_coverage": float(1 - stats.chi2.cdf(lr_cc, 2))}


def backtest(returns: pd.Series, risk: pd.DataFrame, level: float = 0.99) -> dict:
    """Breach counts and tests for a VaR series, plus realised vs predicted ES on ES-tail days."""
    df = pd.concat([returns.rename("r"), risk], axis=1).dropna()
    breaches = df["r"] < -df["var"]
    out = {**kupiec(breaches, level), **christoffersen(breaches, level)}
    tail = df[df["r"] < -df["var"]]
    out["mean_loss_on_breach"] = float(-tail["r"].mean()) if len(tail) else np.nan
    out["mean_es_on_breach"] = float(tail["es"].mean()) if len(tail) else np.nan
    return out


def breaches_by_year(returns: pd.Series, risk: pd.DataFrame) -> pd.Series:
    df = pd.concat([returns.rename("r"), risk["var"]], axis=1).dropna()
    return (df["r"] < -df["var"]).groupby(df.index.year).sum()
