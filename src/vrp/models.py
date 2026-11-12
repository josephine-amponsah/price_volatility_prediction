"""Out-of-sample forecasts of next-month realised volatility.

Every model is run walk-forward: parameters are refitted on an expanding window and each forecast
uses only information available at the close of its own day. A training row is usable on day t only
if its 21-day target window has already ended (row index <= t - 21); otherwise the label would leak.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .features import HORIZON, TRADING_DAYS

OOS_START = "2008-01-02"
REFIT_EVERY = 21
HAR_COLS = ["har_d", "har_w", "har_m"]


def _refit_positions(index: pd.DatetimeIndex, start: str, every: int) -> list[int]:
    first = int(index.searchsorted(pd.Timestamp(start)))
    return list(range(first, len(index), every))


# ---------------------------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------------------------

def naive(feat: pd.DataFrame) -> pd.Series:
    """Random-walk benchmark: next month's volatility equals the last month's."""
    return feat["rv_cc_21"].rename("naive")


def implied(feat: pd.DataFrame) -> pd.Series:
    """The market's forecast: VIX. Biased upwards by the variance risk premium."""
    return feat["vix"].rename("vix")


# ---------------------------------------------------------------------------------------------
# HAR family (log-volatility regressions)
# ---------------------------------------------------------------------------------------------

def walk_forward_ols(feat: pd.DataFrame, cols: list[str], name: str,
                     start: str = OOS_START, every: int = REFIT_EVERY, horizon: int = HORIZON) -> pd.Series:
    """Regress log target on log regressors; refit every `every` days on an expanding window."""
    X_all = sm.add_constant(np.log(feat[cols]))
    y_all = np.log(feat["target"])
    preds = pd.Series(np.nan, index=feat.index, name=name)
    for i in _refit_positions(feat.index, start, every):
        train_end = i - horizon                       # last row whose target is fully observed on day i
        X, y = X_all.iloc[: train_end + 1], y_all.iloc[: train_end + 1]
        ok = y.notna() & X.notna().all(axis=1)
        fit = sm.OLS(y[ok], X[ok]).fit()
        block = X_all.iloc[i: i + every]
        # exp(E[log v] + s^2/2): unbiased level forecast under log-normal errors
        preds.iloc[i: i + every] = np.exp(fit.predict(block) + fit.mse_resid / 2)
    return preds


def har(feat: pd.DataFrame, **kw) -> pd.Series:
    return walk_forward_ols(feat, HAR_COLS, "har", **kw)


def har_vix(feat: pd.DataFrame, **kw) -> pd.Series:
    """HAR with implied volatility added: combines the time-series and market views."""
    return walk_forward_ols(feat, HAR_COLS + ["vix"], "har_vix", **kw)


# ---------------------------------------------------------------------------------------------
# GARCH family
# ---------------------------------------------------------------------------------------------

def garch(feat: pd.DataFrame, kind: str = "gjr", start: str = OOS_START, every: int = REFIT_EVERY,
          horizon: int = HORIZON) -> pd.DataFrame:
    """Walk-forward GARCH(1,1) or GJR-GARCH(1,1,1) with Student-t errors on daily returns (in %).

    Returns a frame with
      * `vol_21`  : annualised volatility forecast over the next `horizon` days (decimal)
      * `sigma_1` : next-day return standard deviation (decimal, daily)
      * `nu`      : fitted Student-t degrees of freedom (for VaR/ES quantiles)
      * `mu`      : fitted daily mean return (decimal)
    """
    from arch import arch_model

    r = feat["ret"].dropna() * 100
    o = 1 if kind == "gjr" else 0
    am = arch_model(r, mean="Constant", vol="GARCH", p=1, o=o, q=1, dist="t", rescale=False)
    out = pd.DataFrame(np.nan, index=r.index, columns=["vol_21", "sigma_1", "nu", "mu"])
    positions = _refit_positions(r.index, start, every)
    for k, i in enumerate(positions):
        last_obs = r.index[i + 1] if i + 1 < len(r) else None     # fit uses data up to and including day i
        res = am.fit(last_obs=last_obs, disp="off", options={"maxiter": 500})
        end = positions[k + 1] if k + 1 < len(positions) else len(r)
        fc = res.forecast(start=r.index[i], horizon=horizon, reindex=False)
        var = fc.variance.loc[r.index[i]: r.index[end - 1]]
        block = out.index[i:end]
        out.loc[block, "vol_21"] = np.sqrt(var.sum(axis=1).values * TRADING_DAYS / horizon) / 100
        out.loc[block, "sigma_1"] = np.sqrt(var.iloc[:, 0].values) / 100
        out.loc[block, "nu"] = res.params["nu"]
        out.loc[block, "mu"] = res.params["mu"] / 100
    return out.reindex(feat.index)


# ---------------------------------------------------------------------------------------------
# LSTM challenger
# ---------------------------------------------------------------------------------------------

LSTM_INPUTS = ["har_d", "har_w", "har_m", "vix"]


def lstm(feat: pd.DataFrame, start: str = OOS_START, seq_len: int = 22, horizon: int = HORIZON,
         epochs: int = 40, seed: int = 0) -> pd.Series:
    """Small LSTM on 22-day sequences of log HAR inputs, log VIX and returns; refitted every January."""
    import torch
    from torch import nn

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(4)

    X_df = np.log(feat[LSTM_INPUTS]).assign(ret=feat["ret"] * 10)
    y_s = np.log(feat["target"])
    X_np, y_np = X_df.to_numpy(np.float32), y_s.to_numpy(np.float32)
    n_feat = X_np.shape[1]

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(n_feat, 32, batch_first=True)
            self.head = nn.Linear(32, 1)

        def forward(self, x):
            h, _ = self.lstm(x)
            return self.head(h[:, -1]).squeeze(-1)

    def windows(end_rows):
        return np.stack([X_np[j - seq_len + 1: j + 1] for j in end_rows])

    preds = pd.Series(np.nan, index=feat.index, name="lstm")
    years = sorted({d.year for d in feat.index[feat.index >= pd.Timestamp(start)]})
    for year in years:
        i = int(feat.index.searchsorted(pd.Timestamp(f"{year}-01-01")))
        block_end = int(feat.index.searchsorted(pd.Timestamp(f"{year + 1}-01-01")))
        train_rows = np.arange(seq_len, i - horizon + 1)
        train_rows = train_rows[np.isfinite(y_np[train_rows]) & np.isfinite(X_np[train_rows]).all(axis=1)]
        mu, sd = np.nanmean(X_np[: i - horizon + 1], 0), np.nanstd(X_np[: i - horizon + 1], 0)
        y_mu, y_sd = y_np[train_rows].mean(), y_np[train_rows].std()
        Xtr = torch.tensor((windows(train_rows) - mu) / sd)
        ytr = torch.tensor((y_np[train_rows] - y_mu) / y_sd)

        n_val = len(train_rows) // 7                            # last ~15% of the window for early stopping
        net, best, best_state = Net(), np.inf, None
        opt = torch.optim.Adam(net.parameters(), lr=3e-3)
        for _ in range(epochs):
            net.train()
            perm = torch.randperm(len(Xtr) - n_val)
            for b in range(0, len(perm), 128):
                idx = perm[b: b + 128]
                opt.zero_grad()
                loss = nn.functional.mse_loss(net(Xtr[idx]), ytr[idx])
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                val = nn.functional.mse_loss(net(Xtr[-n_val:]), ytr[-n_val:]).item()
            if val < best:
                best, best_state = val, {k: v.clone() for k, v in net.state_dict().items()}
        net.load_state_dict(best_state)

        rows = np.arange(i, block_end)
        with torch.no_grad():
            z = net(torch.tensor((windows(rows) - mu) / sd)).numpy()
        preds.iloc[i:block_end] = np.exp(z * y_sd + y_mu + (best * y_sd ** 2) / 2)
    return preds
