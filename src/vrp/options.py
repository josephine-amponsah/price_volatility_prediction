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


def fit_svi(log_moneyness, total_variance, t: float | None = None, weights=None) -> dict:
    """Raw SVI fit that respects the basic no-arbitrage conditions.

    * minimum total variance a + b*sigma*sqrt(1 - rho^2) >= 0 (enforced: `a` is parameterised
      as that minimum, which is bounded at zero)
    * |rho| <= 0.98, so the fit cannot sit on the boundary and turn a wing into a straight line
    * Roger Lee's wing bound b(1 + |rho|) <= 4/t, enforced when `t` (years to expiry) is given
    Several starting points are tried and the best fit is kept.
    """
    k, w = np.asarray(log_moneyness, float), np.asarray(total_variance, float)
    wts = np.ones_like(w) if weights is None else np.asarray(weights, float)
    b_max = 4 / t / 2 if t else 5.0          # Lee bound with |rho| at most 1

    def unpack(p):
        min_var, b, rho, m, sigma = p
        a = min_var - b * sigma * np.sqrt(1 - rho ** 2)
        return a, b, rho, m, sigma

    def resid(p):
        return wts * (svi_total_variance(k, *unpack(p)) - w)

    lb = [0.0, 1e-6, -0.98, -1.0, 1e-3]
    ub = [max(w.max(), 1e-4), b_max, 0.98, 1.0, 2.0]
    best = None
    for rho0 in (-0.7, -0.3, 0.0):
        for sigma0 in (0.05, 0.2):
            x0 = [min(w.min(), ub[0]) * 0.9, min(0.1, b_max / 2), rho0, float(k[np.argmin(w)]), sigma0]
            x0 = np.clip(x0, lb, ub)
            res = optimize.least_squares(resid, x0, bounds=(lb, ub))
            if best is None or res.cost < best.cost:
                best = res
    a, b, rho, m, sigma = (float(x) for x in unpack(best.x))
    return {"a": a, "b": b, "rho": rho, "m": m, "sigma": sigma,
            "rmse": float(np.sqrt(np.mean((svi_total_variance(k, a, b, rho, m, sigma) - w) ** 2))),
            "min_total_variance": float(best.x[0]),
            "lee_wing_ok": bool(t is None or b * (1 + abs(rho)) <= 4 / t + 1e-12)}
