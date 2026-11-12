"""Forecast evaluation: loss functions, Diebold-Mariano, Mincer-Zarnowitz and encompassing tests.

Forecasts of 21-day volatility made on consecutive days overlap by 20 days, so every regression
and test here uses HAC (Newey-West) standard errors with 21 lags.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

HAC = {"cov_type": "HAC", "cov_kwds": {"maxlags": 21}}


def qlike(actual_vol: pd.Series, forecast_vol: pd.Series) -> pd.Series:
    """QLIKE loss on variances (Patton, 2011): robust to noise in the realised-volatility proxy."""
    ratio = actual_vol ** 2 / forecast_vol ** 2
    return ratio - np.log(ratio) - 1


def diebold_mariano(actual: pd.Series, f1: pd.Series, f2: pd.Series, loss: str = "se") -> dict:
    """H0: equal expected loss. Negative statistic means f1 is more accurate than f2."""
    if loss == "se":
        d = (actual - f1) ** 2 - (actual - f2) ** 2
    else:
        d = qlike(actual, f1) - qlike(actual, f2)
    d = d.dropna()
    fit = sm.OLS(d, np.ones(len(d))).fit(**HAC)
    return {"dm_stat": float(fit.tvalues.iloc[0]), "p_value": float(fit.pvalues.iloc[0])}


def mincer_zarnowitz(actual: pd.Series, forecast: pd.Series) -> dict:
    """actual = a + b * forecast. An unbiased forecast has a = 0, b = 1."""
    df = pd.concat([actual, forecast], axis=1).dropna()
    fit = sm.OLS(df.iloc[:, 0], sm.add_constant(df.iloc[:, 1])).fit(**HAC)
    wald = fit.wald_test((np.eye(2), np.array([0.0, 1.0])), scalar=True)
    return {"alpha": float(fit.params.iloc[0]), "beta": float(fit.params.iloc[1]),
            "r2": float(fit.rsquared), "p_unbiased": float(wald.pvalue)}


def encompassing(actual: pd.Series, forecasts: dict[str, pd.Series]) -> pd.DataFrame:
    """Regress realised volatility on several forecasts at once (HAC t-stats).

    A forecast whose coefficient is significant carries information the others do not.
    """
    df = pd.concat([actual.rename("actual"), *[s.rename(k) for k, s in forecasts.items()]], axis=1).dropna()
    fit = sm.OLS(df["actual"], sm.add_constant(df.drop(columns="actual"))).fit(**HAC)
    return pd.DataFrame({"coef": fit.params, "t_stat": fit.tvalues, "p_value": fit.pvalues})


def score_table(actual: pd.Series, forecasts: dict[str, pd.Series], benchmark: str = "vix") -> pd.DataFrame:
    """One row per model: RMSE and MAE in vol points, QLIKE, Mincer-Zarnowitz R^2, DM test vs benchmark."""
    rows = {}
    for name, f in forecasts.items():
        df = pd.concat([actual, f], axis=1).dropna()
        a, p = df.iloc[:, 0], df.iloc[:, 1]
        mz = mincer_zarnowitz(a, p)
        row = {
            "rmse_vol_pts": float(np.sqrt(((a - p) ** 2).mean()) * 100),
            "mae_vol_pts": float((a - p).abs().mean() * 100),
            "qlike": float(qlike(a, p).mean()),
            "bias_vol_pts": float((p - a).mean() * 100),
            "mz_r2": mz["r2"],
        }
        if name != benchmark:
            dm = diebold_mariano(actual, f, forecasts[benchmark], loss="qlike")
            row["dm_qlike_vs_" + benchmark] = dm["dm_stat"]
            row["dm_p"] = dm["p_value"]
        rows[name] = row
    return pd.DataFrame(rows).T
