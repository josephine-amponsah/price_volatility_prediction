"""Snapshot the SPY option chain closest to 30 days, fit an SVI smile and compare it with the forecasts.

    python scripts/fetch_option_chain.py                 # expiry nearest 30 calendar days
    python scripts/fetch_option_chain.py --target-days 60

Writes data/options/spy_chain_<date>_<expiry>.csv, reports/svi_fit.json and reports/figures/smile.png.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vrp import options  # noqa: E402


def pick_expiry(expiries: list[str], target_days: int) -> str:
    today = pd.Timestamp(date.today())
    days = [(pd.Timestamp(e) - today).days for e in expiries]
    eligible = [(abs(d - target_days), e) for d, e in zip(days, expiries) if d >= 7]
    return min(eligible)[1]


def main(target_days: int, rate: float, max_abs_k: float) -> None:
    import yfinance as yf

    tk = yf.Ticker("SPY")
    expiry = pick_expiry(list(tk.options), target_days)
    spot = float(tk.history(period="1d")["Close"].iloc[-1])
    chain = tk.option_chain(expiry)
    t = (pd.Timestamp(expiry) + pd.Timedelta(hours=16) - pd.Timestamp.now()).total_seconds() / (365 * 86400)
    forward = spot * np.exp(rate * t)

    # out-of-the-money options only (puts below the forward, calls above), with two-sided quotes
    puts = chain.puts[chain.puts.strike < forward].assign(call=False)
    calls = chain.calls[chain.calls.strike >= forward].assign(call=True)
    df = pd.concat([puts, calls])
    df = df[(df.bid > 0) & (df.ask > df.bid) & (df.openInterest.fillna(0) > 0)].copy()
    df["mid"] = (df.bid + df.ask) / 2
    df["iv"] = [options.implied_vol(m, spot, k, t, rate, c) for m, k, c in zip(df.mid, df.strike, df.call)]
    df["log_moneyness"] = np.log(df.strike / forward)
    df = df.dropna(subset=["iv"])
    df = df[df.log_moneyness.abs() <= max_abs_k].sort_values("strike")
    df["total_variance"] = df.iv ** 2 * t

    out = ROOT / "data" / "options"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"spy_chain_{date.today()}_{expiry}.csv", index=False)

    fit = options.fit_svi(df.log_moneyness, df.total_variance, t=t)
    p = (fit["a"], fit["b"], fit["rho"], fit["m"], fit["sigma"])
    atm = np.sqrt(options.svi_total_variance(0.0, *p) / t)
    iv95 = np.sqrt(options.svi_total_variance(np.log(0.95), *p) / t)
    iv105 = np.sqrt(options.svi_total_variance(np.log(1.05), *p) / t)

    summary = {"date": str(date.today()), "expiry": expiry, "days": round(t * 365, 1), "spot": spot,
               "quotes": len(df), "svi": {k: round(v, 5) if isinstance(v, float) else v for k, v in fit.items()},
               "atm_iv": round(float(atm), 4), "iv_95_strike": round(float(iv95), 4),
               "iv_105_strike": round(float(iv105), 4), "skew_95_minus_105": round(float(iv95 - iv105), 4)}
    fc_path = ROOT / "build" / "forecasts.parquet"
    if fc_path.exists():
        last = pd.read_parquet(fc_path).drop(columns="target").ffill().iloc[-1]
        summary["latest_forecasts"] = {k: round(float(v), 4) for k, v in last.items()}
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "svi_fit.json").write_text(json.dumps(summary, indent=2))

    grid = np.linspace(df.log_moneyness.min(), df.log_moneyness.max(), 200)
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    ax.scatter(df.log_moneyness, df.iv * 100, s=10, color="#7f7f7f", label="Market (mid)")
    ax.plot(grid, np.sqrt(options.svi_total_variance(grid, *p) / t) * 100, color="#1f4e79", label="SVI fit")
    ax.axvline(0, color="black", lw=0.5)
    ax.set_xlabel("Log-moneyness ln(K/F)")
    ax.set_ylabel("Implied volatility (%)")
    ax.set_title(f"SPY smile, {expiry} expiry ({t * 365:.0f} days), as of {date.today()}", loc="left")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(ROOT / "reports" / "figures" / "smile.png")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-days", type=int, default=30)
    ap.add_argument("--rate", type=float, default=0.04, help="continuously compounded risk-free rate")
    ap.add_argument("--max-abs-k", type=float, default=0.25, help="drop strikes beyond this |ln(K/F)|")
    a = ap.parse_args()
    main(a.target_days, a.rate, a.max_abs_k)
