"""Load SPY daily prices and the VIX index into one aligned frame.

The raw CSVs live in data/raw/ and are committed so results reproduce without network access.
`refresh()` re-downloads them with yfinance when you want newer data.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"


def load(raw_dir: Path = RAW) -> pd.DataFrame:
    """Return SPY OHLC + adjusted close and the VIX close, on dates where both exist."""
    spy = pd.read_csv(raw_dir / "spy_daily.csv", parse_dates=["date"], index_col="date")
    vix = pd.read_csv(raw_dir / "vix_daily.csv", parse_dates=["DATE"], index_col="DATE")
    vix = vix.rename(columns=str.lower)[["close"]].rename(columns={"close": "vix"})
    df = spy.join(vix, how="inner").sort_index()
    df = df[(df[["open", "high", "low", "close", "adj_close", "vix"]] > 0).all(axis=1)]
    df.index.name = "date"
    return df


def refresh(start: str = "2003-01-01", raw_dir: Path = RAW) -> None:  # pragma: no cover - network
    """Download fresh SPY and VIX history with yfinance and overwrite data/raw."""
    import yfinance as yf

    spy = yf.download("SPY", start=start, auto_adjust=False, progress=False)
    spy.columns = spy.columns.get_level_values(0)
    spy = spy.rename(columns={"Open": "open", "High": "high", "Low": "low",
                              "Close": "close", "Adj Close": "adj_close"})
    spy.index.name = "date"
    spy[["open", "high", "low", "close", "adj_close"]].to_csv(raw_dir / "spy_daily.csv")

    vix = yf.download("^VIX", start=start, auto_adjust=False, progress=False)
    vix.columns = vix.columns.get_level_values(0)
    vix = vix.rename(columns=str.upper)[["OPEN", "HIGH", "LOW", "CLOSE"]]
    vix.index.name = "DATE"
    vix.to_csv(raw_dir / "vix_daily.csv")
