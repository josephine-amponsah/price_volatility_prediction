# Volatility Risk Premium Forecasting for Derivatives Trading Desks

## Problem Statement

Options prices embed an implied volatility (IV) that reflects the market's consensus expectation
of future realised volatility (RV). The systematic gap between IV and actual RV — the Volatility
Risk Premium (VRP) — represents a persistent cost to options buyers and a persistent revenue
source for options sellers. However, the VRP is not constant: it compresses, inverts, and spikes
across market regimes.

A derivatives desk using only IV has no ex-ante view on whether the VRP will be positive or
negative over the coming expiry window. This project builds an independent statistical forecast
of 30-day realised volatility (RV) for SPY, and uses the divergence between forecasted RV and
prevailing IV to generate actionable trading signals for the derivatives desk.

## Value Proposition

| Signal | Condition | Desk Action |
|---|---|---|
| SELL VOL | Forecast RV << IV | Options are expensive — collect the VRP |
| NEUTRAL | Forecast RV ≈ IV | No material edge — hold |
| BUY VOL | Forecast RV >> IV | Options are cheap — pay for optionality |

## Project Structure

```
price_volatility_prediction/
├── data/
│   └── processed_vol_features.csv   # Engineered features (SPY 2020–2025)
├── notebooks/
│   ├── eda_price_volatility.ipynb   # Feature engineering & exploratory analysis
│   └── volatility_forecasting.ipynb # Model training, evaluation & signal generation
├── requirements.txt
└── README.md
```

## Data

- **Underlying**: SPY (S&P 500 ETF), daily OHLCV, 2020–2025
- **IV Proxy**: VIX (CBOE Volatility Index), daily close
- **Source**: `yfinance`

## Features Engineered

| Feature | Description |
|---|---|
| `realized_vol5/10/21` | Rolling close-to-close RV (annualised) |
| `parkinson` | High-Low range estimator (more efficient than C2C) |
| `gk_vol` | Garman-Klass OHLC estimator |
| `rv_lag1/5/21` | HAR-RV lags (daily, weekly, monthly) |
| `vix` / `vix_change` | Implied volatility level and daily change |
| `mom_5d` / `mom_21d` | Price momentum (5-day, 21-day) |
| `vol_of_vol5` | Volatility of volatility (5-day) |
| **Target** | `realized_vol21` (30-day RV) over next expiry window |

## Models

| Model | Type | Notes |
|---|---|---|
| ARIMA(2,0,1) | Univariate baseline | Walk-forward, 1-step |
| HAR-RV | Multivariate baseline | OLS on daily/weekly/monthly RV lags |
| MLP | Deep learning | Sequence flattening, 2 hidden layers |
| LSTM | Deep learning | Stacked, 2 layers, sequence input |

## Evaluation Framework

### 1. Forecast Accuracy

| Metric | Description |
|---|---|
| MAPE | Mean absolute % error vs actual RV |
| RMSE | Root mean squared error |
| Directional Accuracy | % of correct UP/DOWN calls |
| Mean Bias | Systematic over/under-forecasting |
| MZ R² | Mincer-Zarnowitz calibration (slope, intercept, R²) |

### 2. Trading Signal Quality (vs IV)

| Metric | Description |
|---|---|
| VRP Signal | `IV − Forecast RV` per expiry date |
| Signal Hit Rate | % of SELL/BUY signals where model was correct |
| Encompassing Regression | Does the model add information beyond IV alone? |
| Regime Breakdown | Signal quality across low / mid / high VIX regimes |

## Key Results

*(To be populated after full evaluation)*

| Model | MAPE | Dir Acc | Signal Hit Rate |
|---|---|---|---|
| ARIMA(2,0,1) | 7.07% | 53.95% | — |
| HAR-RV | — | — | — |
| MLP | 50.50% | 54.10% | — |
| LSTM | — | — | — |

## Setup

```bash
git clone https://github.com/josephine-amponsah/price_volatility_prediction.git
cd price_volatility_prediction
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run notebooks in order:
1. `notebooks/eda_price_volatility.ipynb` — feature engineering & exploratory analysis
2. `notebooks/volatility_forecasting.ipynb` — model training, evaluation & signal generation

## Requirements

```
pandas, numpy, matplotlib, scikit-learn, torch, statsmodels, yfinance, arch, scipy
```

## Audience

This project is designed for **derivatives trading desks** seeking an independent,
statistically-grounded view on volatility mis-pricing — complementary to, and
benchmarked against, market-implied volatility (IV/VIX).