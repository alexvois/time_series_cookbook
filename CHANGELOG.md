# Changelog

## 2025-10-05 — Standardize outputs and prediction logic across scripts

This change harmonizes scaling, printed output, metrics, and plotting across the ARIMA, XGBoost, and PyTorch time series examples.

### ARIMA (time_series_arima1.py)
- Trend-aware model selection by AIC: searches (p,d,q) with trend ∈ {"n","c"}.
- Optional log-space modeling (USE_LOG=True by default), with inversion for outputs.
- Rolling 1-step-ahead predictions on the test range (uses true next observation at each step).
- Added Predicted Train Data line (in-sample fitted values), and Train/Test MSE/MAE.
- Standardized plot labels: "Original Train Data", "Predicted Train Data", "Original Test Data", "Predicted Test Data".

### XGBoost (time_series_xgboost1.py)
- Unified predictions message: "Making predictions (sliding window, 1-step ahead)...".
- Kept consistent plot labels; retains delta-target approach (predicts delta to last value in window).
- Prints Train/Test MSE and MAE.

### PyTorch (time_series_pytorch1.py)
- Scaler now fit on train range only (including the seq_length context used by training windows), then applied to the full series.
- Unified predictions message: "Making predictions (sliding window, 1-step ahead)...".
- Prints Train/Test MSE and MAE on scaled values.

### Consistency
- Print structure: sample counts, training info, unified predictions message, Train/Test metrics.
- Plot labels match across all three scripts.

### Notes / Potential follow-ups
- ARIMA currently splits on raw points; PyTorch/XGBoost split on sequences (seq_length=50). If preferred, ARIMA can be adjusted to use a sequence-aware split so all three show identical test sample counts (e.g., 209).
- Add flags to toggle log-space modeling (ARIMA) and static multi-step vs rolling 1-step predictions for quick comparisons.

Reference commit: 81bd83b
