"""Run the whole study: forecasts -> evaluation -> strategy backtest -> risk backtest -> figures.

    python scripts/run_all.py            # full run (LSTM included, ~3 minutes on a laptop CPU)
    python scripts/run_all.py --fast     # skip the LSTM (used by CI)

Outputs: reports/results.json, reports/tables.md, reports/figures/*.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vrp import backtest as bt  # noqa: E402
from vrp import data, evaluate, features, models, risk  # noqa: E402

REPORTS, FIGS, BUILD = ROOT / "reports", ROOT / "reports" / "figures", ROOT / "build"
COST = 0.5          # vol points of vega per swap opened
THRESHOLDS = [0, 2, 4]
SIGNAL_MODELS = ["har", "har_vix", "gjr_garch_t"]
NAMES = {"naive": "Naive (last 21 days)", "vix": "VIX (implied)", "har": "HAR-RV",
         "har_vix": "HAR-RV + VIX", "garch_t": "GARCH(1,1)-t", "gjr_garch_t": "GJR-GARCH-t", "lstm": "LSTM"}


def forecasts(feat: pd.DataFrame, fast: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    BUILD.mkdir(exist_ok=True)
    g = models.garch(feat, "garch").add_prefix("garch_")
    gj = models.garch(feat, "gjr").add_prefix("gjr_")
    fc = {"naive": models.naive(feat), "vix": models.implied(feat), "har": models.har(feat),
          "har_vix": models.har_vix(feat), "garch_t": g["garch_vol_21"], "gjr_garch_t": gj["gjr_vol_21"]}
    if not fast:
        fc["lstm"] = models.lstm(feat)
    F = pd.DataFrame(fc).assign(target=feat["target"])
    G = g.join(gj)
    F.to_parquet(BUILD / "forecasts.parquet")
    G.to_parquet(BUILD / "garch.parquet")
    return F, G


def signal(feat, F, model, thr):
    return pd.Series(np.where((feat["vix"] - F[model]) * 100 > thr, -1.0, 0.0), index=feat.index)


def strategies(feat, F, offset=0):
    start = feat.index[feat.index.searchsorted(pd.Timestamp(models.OOS_START)) + offset]
    out = {"always_short": bt.variance_swap_pnl(feat, pd.Series(-1.0, index=feat.index), COST, start=start)}
    for m in SIGNAL_MODELS:
        for thr in THRESHOLDS:
            out[f"{m}>{thr}"] = bt.variance_swap_pnl(feat, signal(feat, F, m, thr), COST, start=start)
    return out


def md_table(df: pd.DataFrame, floatfmt: str = "{:.2f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join([df.index.name or ""] + cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for idx, row in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in row]
        lines.append("| " + " | ".join([str(idx)] + cells) + " |")
    return "\n".join(lines)


def main(fast: bool) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    feat = features.build(data.load())
    F, G = forecasts(feat, fast)
    oos = F.loc[models.OOS_START:].dropna()
    results: dict = {"sample": {"start": str(feat.index[0].date()), "end": str(feat.index[-1].date()),
                                "oos_start": str(oos.index[0].date()), "oos_end": str(oos.index[-1].date()),
                                "oos_days": len(oos)}}

    # ---- 1. forecast accuracy -------------------------------------------------------------
    model_cols = [c for c in F.columns if c != "target"]
    scores = evaluate.score_table(oos["target"], {k: oos[k] for k in model_cols})
    enc = {m: evaluate.encompassing(oos["target"], {"vix": oos["vix"], m: oos[m]})
           for m in model_cols if m not in ("vix",)}
    vrp = (oos["vix"] - oos["target"]) * 100
    results["forecasts"] = scores.to_dict(orient="index")
    results["encompassing"] = {m: e.to_dict(orient="index") for m, e in enc.items()}
    results["vrp"] = {"mean_vol_pts": float(vrp.mean()), "median_vol_pts": float(vrp.median()),
                      "share_days_positive": float((vrp > 0).mean())}

    # ---- 2. strategy ----------------------------------------------------------------------
    strat = strategies(feat, F)
    summ = pd.DataFrame({k: bt.summary(v["pnl"]) for k, v in strat.items()}).T
    monthly = {k: v["pnl"].groupby(np.arange(len(v)) // 21).sum() for k, v in strat.items()}
    trial_sr = [m.mean() / m.std() for m in monthly.values()]
    best = summ["sharpe"].drop("always_short").idxmax()
    dsr = bt.deflated_sharpe(monthly[best], trial_sr)
    # robustness: the same rules started on each of the 21 possible roll days
    offs = pd.DataFrame({o: {k: bt.summary(v["pnl"])["sharpe"] for k, v in strategies(feat, F, o).items()}
                         for o in range(21)})
    results["strategy"] = summ.to_dict(orient="index")
    results["strategy_best"] = {"name": best, **dsr}
    results["roll_offset_sharpe"] = {k: {"min": float(r.min()), "median": float(r.median()), "max": float(r.max())}
                                     for k, r in offs.iterrows()}
    yearly = pd.DataFrame({k: strat[k]["pnl"].groupby(strat[k].index.year).sum()
                           for k in ["always_short", best]})
    results["yearly_pnl"] = yearly.to_dict()

    # ---- 3. market risk -------------------------------------------------------------------
    r = feat["ret"].loc[models.OOS_START:]
    # the GARCH row dated d is the forecast for day d+1, so shift it onto the day it applies to
    garch_risk = risk.student_t_var_es(G["gjr_mu"], G["gjr_sigma_1"], G["gjr_nu"]).shift(1).loc[r.index]
    hs_risk = risk.historical_var_es(feat["ret"]).loc[r.index]
    results["var_spy"] = {"gjr_garch_t": risk.backtest(r, garch_risk), "historical_250d": risk.backtest(r, hs_risk)}
    by_year = pd.DataFrame({"GJR-GARCH-t": risk.breaches_by_year(r, garch_risk),
                            "Historical 250d": risk.breaches_by_year(r, hs_risk)})
    results["var_breaches_by_year"] = by_year.to_dict()
    pnl = strat[best]["pnl"]
    strat_hs = risk.historical_var_es(pnl).loc[pnl.index]
    results["var_strategy_hs"] = risk.backtest(pnl, strat_hs)

    (REPORTS / "results.json").write_text(json.dumps(results, indent=2, default=float))

    # ---- tables ---------------------------------------------------------------------------
    t1 = scores.rename(index=NAMES)[["rmse_vol_pts", "qlike", "bias_vol_pts", "mz_r2"]].copy()
    t1["DM vs VIX (QLIKE) p"] = scores.get("dm_p").rename(index=NAMES)
    t1.index.name = "Forecast"
    t2 = summ[["sharpe", "max_drawdown", "worst_month", "hit_rate_months", "share_months_invested"]].copy()
    t2["sharpe_median_21_roll_days"] = [offs.loc[k].median() for k in t2.index]
    t2["sharpe_range_21_roll_days"] = [f"{offs.loc[k].min():.2f} to {offs.loc[k].max():.2f}" for k in t2.index]
    t2.index.name = "Strategy"
    (REPORTS / "tables.md").write_text(
        "## Forecasts (out of sample)\n\n" + md_table(t1) +
        "\n\n## Strategies (vol points per unit vega, after costs)\n\n" + md_table(t2) +
        "\n\n## VaR breaches by year\n\n" + md_table(by_year.rename_axis("Year"), "{:.0f}") + "\n")

    # ---- figures --------------------------------------------------------------------------
    plt.rcParams.update({"figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.3, "font.size": 9})
    c_main, c_alt, c_grey = "#1f4e79", "#c0504d", "#7f7f7f"

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(strat["always_short"]["pnl"].cumsum(), color=c_grey, lw=1.2,
            label=f"Always short (Sharpe {summ.loc['always_short', 'sharpe']:.2f})")
    ax.plot(strat[best]["pnl"].cumsum(), color=c_main, lw=1.6,
            label=f"Short only when VIX exceeds the {NAMES[best.split('>')[0]]} forecast by "
                  f"> {best.split('>')[1]} vol pts (Sharpe {summ.loc[best, 'sharpe']:.2f})")
    ax.set_ylabel("Cumulative P&L (vol points per unit vega)")
    ax.set_title("Selling one-month SPY variance, 2008–2026, after 0.5 vol pt cost per trade", loc="left")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGS / "equity_curve.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(oos.index, oos["target"] * 100, color=c_grey, lw=0.8, label="Realised (next 21 days)")
    ax.plot(oos.index, oos["vix"] * 100, color=c_alt, lw=0.8, label="VIX")
    ax.plot(oos.index, oos["har_vix"] * 100, color=c_main, lw=0.8, label="HAR-RV + VIX forecast")
    ax.set_yscale("log")
    ax.set_ylabel("Annualised volatility (%)")
    ax.set_title("Forecasts of next-month SPY volatility (out of sample)", loc="left")
    ax.legend(frameon=False, ncol=3, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGS / "forecasts.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 3.2))
    roll = vrp.rolling(63).mean()
    ax.fill_between(vrp.index, vrp, 0, where=vrp >= 0, color=c_main, alpha=0.35, lw=0)
    ax.fill_between(vrp.index, vrp, 0, where=vrp < 0, color=c_alt, alpha=0.6, lw=0)
    ax.plot(roll.index, roll, color="black", lw=0.9, label="3-month average")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("VIX − realised (vol points)")
    ax.set_title(f"The premium: VIX exceeded next-month realised volatility on {(vrp > 0).mean():.0%} of days",
                 loc="left")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGS / "vrp.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(r.index, r * 100, color=c_grey, lw=0.4)
    ax.plot(garch_risk.index, -garch_risk["var"] * 100, color=c_main, lw=0.8, label="99% VaR, GJR-GARCH-t")
    ax.plot(hs_risk.index, -hs_risk["var"] * 100, color=c_alt, lw=0.8, label="99% VaR, historical 250d")
    br = r < -garch_risk["var"]
    ax.scatter(r.index[br], r[br] * 100, s=8, color=c_main, zorder=3)
    br2 = r < -hs_risk["var"]
    ax.scatter(r.index[br2], r[br2] * 100, s=8, color=c_alt, marker="x", zorder=3)
    ax.set_ylabel("Daily SPY return (%)")
    ax.set_title("One-day 99% VaR and its breaches", loc="left")
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIGS / "var_backtest.png")
    plt.close(fig)

    print(json.dumps({k: results[k] for k in ["sample", "vrp", "strategy_best"]}, indent=2, default=float))
    print((REPORTS / "tables.md").read_text())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fast", action="store_true", help="skip the LSTM")
    main(p.parse_args().fast)
