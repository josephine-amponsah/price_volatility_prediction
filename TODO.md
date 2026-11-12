## TO-DO

Priority 1 — Fix existing bugs before adding anything new
  └─ Fix vol_of_vol5 double-annualisation
  └─ Fix gk_vol formula
  └─ Exclude target_rv from feature_cols
  └─ Switch MinMaxScaler → StandardScaler
  └─ Align target to realized_vol21.shift(-21)

Priority 2 — Complete the model comparison
  └─ Add HAR-RV baseline (OLS, 10 lines)
  └─ Evaluate all LSTM versions side-by-side
  └─ Add bias and MZ R² to evaluate_model()
  └─ Plot train/val loss curves

Priority 3 — Add the IV comparison (core deliverable)
  └─ Compute IV error vs Model error per date
  └─ Run encompassing regression
  └─ Build VRP signal table (SELL/BUY/NEUTRAL)
  └─ Compute signal hit rate

Priority 4 — Add regime analysis
  └─ Segment test results by VIX level (low/mid/high)
  └─ Report metrics per regime

Priority 5 — Robustness
  └─ Walk-forward cross-validation (21-day rolling windows)
  └─ Add VIX term structure slope as a feature
  └─ Add Yang-Zhang estimator
