import numpy as np
import pandas as pd
import pytest

from vrp import backtest as bt
from vrp import features, options, risk


def _prices(n=300, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    open_ = close * np.exp(rng.normal(0, 0.002, n))
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.01, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.01, n))
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                         "adj_close": close, "vix": 20.0}, index=idx)


def test_target_has_no_lookahead_into_features():
    """Changing a future price must not change any feature on earlier days."""
    df = _prices()
    base = features.build(df)
    shocked = df.copy()
    shocked.iloc[200, :5] *= 1.10
    after = features.build(shocked)
    cols = [c for c in base.columns if c != "target"]
    day = df.index[200]
    pd.testing.assert_frame_equal(base.loc[: day - pd.Timedelta(days=1), cols],
                                  after.loc[: day - pd.Timedelta(days=1), cols])


def test_target_is_forward_window():
    r = pd.Series(np.r_[np.zeros(30), np.full(21, 0.01), np.zeros(30)])
    target = features.forward_realised_vol(r, 21)
    assert target.iloc[29] == pytest.approx(0.01 * np.sqrt(252))     # covers rows 30..50 exactly
    assert target.iloc[50] == pytest.approx(0.0)


def test_garman_klass_zero_range_is_zero():
    s = pd.Series([100.0, 100.0])
    assert (features.garman_klass_var(s, s, s, s) == 0).all()


def test_put_call_parity():
    s, k, t, r, v = 100, 95, 0.5, 0.03, 0.25
    c = options.bs_price(s, k, t, r, v, call=True)
    p = options.bs_price(s, k, t, r, v, call=False)
    assert c - p == pytest.approx(s - k * np.exp(-r * t), abs=1e-10)


def test_implied_vol_round_trip():
    price = options.bs_price(100, 110, 0.25, 0.01, 0.32, call=True)
    assert options.implied_vol(price, 100, 110, 0.25, 0.01, call=True) == pytest.approx(0.32, abs=1e-8)


def test_svi_recovers_parameters():
    k = np.linspace(-0.4, 0.3, 25)
    true = {"a": 0.01, "b": 0.12, "rho": -0.6, "m": 0.02, "sigma": 0.15}
    fit = options.fit_svi(k, options.svi_total_variance(k, **true), t=0.25)
    assert fit["rmse"] < 1e-6
    assert fit["min_total_variance"] >= 0 and fit["lee_wing_ok"]
    assert fit["rho"] == pytest.approx(-0.6, abs=1e-3)


def test_svi_respects_no_arbitrage_on_steep_short_dated_skew():
    """A noisy, steep 5-day put skew used to push rho to -1 and the minimum variance below zero."""
    rng = np.random.default_rng(1)
    t = 5 / 365
    k = np.linspace(-0.2, 0.05, 60)
    iv = 0.15 - 1.4 * k + 4 * k ** 2 + rng.normal(0, 0.004, k.size)
    fit = options.fit_svi(k, iv ** 2 * t, t=t)
    assert fit["min_total_variance"] >= 0 and fit["lee_wing_ok"] and abs(fit["rho"]) <= 0.98


def test_kupiec_exact_rate_has_p_value_one():
    b = pd.Series([1] * 10 + [0] * 990)
    assert risk.kupiec(b, 0.99)["p_value"] == pytest.approx(1.0)


def test_student_t_var_matches_quantile():
    mu, sig, nu = pd.Series([0.0]), pd.Series([0.01]), pd.Series([5.0])
    out = risk.student_t_var_es(mu, sig, nu, var_level=0.99)
    from scipy import stats
    expected = -stats.t.ppf(0.01, 5) * np.sqrt(3 / 5) * 0.01
    assert out["var"].iloc[0] == pytest.approx(expected)
    assert out["es"].iloc[0] > 0


def test_short_variance_earns_strike_minus_realised():
    """With constant VIX of 20 and zero returns, a short swap earns K/2 per period before costs."""
    idx = pd.bdate_range("2008-01-01", periods=60)
    feat = pd.DataFrame({"vix": 0.20, "ret": 0.0}, index=idx)
    pnl = bt.variance_swap_pnl(feat, pd.Series(-1.0, index=idx), cost_vol_pts=0.0, start="2008-01-01")
    first_period = pnl["pnl"].iloc[:21].sum()
    assert first_period == pytest.approx(20 ** 2 / (2 * 20))
