"""Black-Scholes pricing, Greeks, implied volatility and an SVI smile fit.

Used to read an option chain: back out implied volatility per strike, fit Gatheral's raw SVI
parameterisation of total implied variance, and compare at-the-money implied volatility with the
model forecasts. `scripts/fetch_option_chain.py` saves a chain snapshot to data/options/.
"""
from __future__ import annotations

import numpy as np
from scipy import optimize, stats

N, n = stats.norm.cdf, stats.norm.pdf


def _d1_d2(s, k, t, r, q, vol):
    d1 = (np.log(s / k) + (r - q + 0.5 * vol ** 2) * t) / (vol * np.sqrt(t))
    return d1, d1 - vol * np.sqrt(t)


def bs_price(s, k, t, r, vol, call=True, q=0.0):
    d1, d2 = _d1_d2(s, k, t, r, q, vol)
    if call:
        return s * np.exp(-q * t) * N(d1) - k * np.exp(-r * t) * N(d2)
    return k * np.exp(-r * t) * N(-d2) - s * np.exp(-q * t) * N(-d1)


def greeks(s, k, t, r, vol, call=True, q=0.0) -> dict:
    """Delta, gamma, vega (per 1 vol point), theta (per calendar day)."""
    d1, d2 = _d1_d2(s, k, t, r, q, vol)
    disc_q, disc_r = np.exp(-q * t), np.exp(-r * t)
    delta = disc_q * N(d1) if call else disc_q * (N(d1) - 1)
    gamma = disc_q * n(d1) / (s * vol * np.sqrt(t))
    vega = s * disc_q * n(d1) * np.sqrt(t) / 100
    common = -s * disc_q * n(d1) * vol / (2 * np.sqrt(t))
    if call:
        theta = common - r * k * disc_r * N(d2) + q * s * disc_q * N(d1)
    else:
        theta = common + r * k * disc_r * N(-d2) - q * s * disc_q * N(-d1)
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta / 365}


def implied_vol(price, s, k, t, r, call=True, q=0.0) -> float:
    """Implied volatility by Brent's method; NaN when the price is outside no-arbitrage bounds."""
    intrinsic = max(s * np.exp(-q * t) - k * np.exp(-r * t), 0) if call else \
        max(k * np.exp(-r * t) - s * np.exp(-q * t), 0)
    if price <= intrinsic + 1e-10:
        return np.nan
    try:
        return optimize.brentq(lambda v: bs_price(s, k, t, r, v, call, q) - price, 1e-4, 5.0, xtol=1e-10)
    except ValueError:
        return np.nan


def svi_total_variance(k, a, b, rho, m, sigma):
    """Raw SVI (Gatheral, 2004): w(k) = a + b (rho (k - m) + sqrt((k - m)^2 + sigma^2)), k = ln(K/F)."""
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))


def fit_svi(log_moneyness, total_variance, weights=None) -> dict:
    """Least-squares raw SVI fit with the usual parameter bounds (b >= 0, |rho| < 1, sigma > 0)."""
    k, w = np.asarray(log_moneyness, float), np.asarray(total_variance, float)
    wts = np.ones_like(w) if weights is None else np.asarray(weights, float)

    def resid(p):
        return wts * (svi_total_variance(k, *p) - w)

    x0 = [max(w.min() * 0.5, 1e-4), 0.1, -0.5, 0.0, 0.1]
    lb, ub = [-1.0, 0.0, -0.999, -1.0, 1e-4], [1.0, 5.0, 0.999, 1.0, 2.0]
    res = optimize.least_squares(resid, x0, bounds=(lb, ub))
    a, b, rho, m, sigma = res.x
    return {"a": a, "b": b, "rho": rho, "m": m, "sigma": sigma,
            "rmse": float(np.sqrt(np.mean((svi_total_variance(k, *res.x) - w) ** 2))),
            "min_variance_nonnegative": bool(a + b * sigma * np.sqrt(1 - rho ** 2) >= 0)}
