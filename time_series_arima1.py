# -*- coding: utf-8 -*-
"""time_series_arima1.py

ARIMA implementation for time series forecasting
Aligned with variable notation and structure from time_series_pytorch1.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA

# Load Stock Market Data from Yahoo Finance

def load_stock_data():
    return yf.download('AAPL', start='2010-01-01', end='2022-02-26')


def get_raw_series():
    df = load_stock_data()
    time_series = df['Close'].values  # Use Close price as time series
    return time_series, df.index


# Set random seed for reproducibility
np.random.seed(0)

# Data preparation (mirrors other scripts)
# Load raw (unscaled) series and date index
data_raw, date_index = get_raw_series()

# Determine split sizes consistently with other examples
DEFAULT_TRAIN_LENGTH = 2800
if len(data_raw) <= DEFAULT_TRAIN_LENGTH + 1:
    # Ensure we have at least some test samples
    train_length = max(1, len(data_raw) - 50)
else:
    train_length = DEFAULT_TRAIN_LENGTH

print()
print(f"created {len(data_raw)} samples")
print(f"train samples: {train_length}")
print(f"test samples: {len(data_raw) - train_length}")

# Split
train_series = data_raw[:train_length]
test_series = data_raw[train_length:]

# Optional: model in log-space to stabilize variance, invert for plotting/metrics
USE_LOG = True
_EPS = 1e-8
train_series_model = np.log(train_series + _EPS) if USE_LOG else train_series
# Note: we do NOT need test_series_model unless doing rolling refits; kept for clarity

# Fit scaler on TRAIN RANGE ONLY to avoid leakage (for metrics/plots)
_scaler = MinMaxScaler().fit(train_series.reshape(-1, 1))

# For metrics and plotting, scale both train and test using train-fitted scaler
data_scaled = _scaler.transform(data_raw.reshape(-1, 1)).flatten()
train_scaled = data_scaled[:train_length]
test_scaled = data_scaled[train_length:]

# Trend-aware ARIMA order selection using AIC on a small grid
candidate_specs = []
for d in (0, 1):
    for p in (0, 1, 2):
        for q in (0, 1, 2):
            # Include constant; for d=1, this becomes a drift term in levels
            trends = ("n", "c")
            for trend in trends:
                candidate_specs.append(((p, d, q), trend))

best_order = None
best_trend = None
best_aic = np.inf
best_model = None

print("Selecting ARIMA order and trend by AIC (small grid)...")
for order, trend in candidate_specs:
    try:
        model = ARIMA(
            train_series_model,
            order=order,
            trend=trend,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        result = model.fit(method_kwargs={"warn_convergence": False})
        if result.aic < best_aic:
            best_aic = result.aic
            best_order = order
            best_trend = trend
            best_model = result
    except Exception:
        # Some specs may fail to converge; skip
        continue

# Fallback if all specs fail (very unlikely)
if best_model is None:
    best_order = (1, 1, 1)
    best_trend = "c"
    best_model = ARIMA(
        train_series_model,
        order=best_order,
        trend=best_trend,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(method_kwargs={"warn_convergence": False})

print(f"Selected ARIMA order: {best_order} trend: '{best_trend}' (AIC={best_aic:.2f})")

# In-sample one-step-ahead fitted values for training range
in_sample_model_space = np.asarray(best_model.fittedvalues)
if USE_LOG:
    predicted_train = np.exp(in_sample_model_space) - _EPS
else:
    predicted_train = in_sample_model_space
predicted_train_scaled = _scaler.transform(predicted_train.reshape(-1, 1)).flatten()

# Forecast next steps equal to test length
steps = len(test_series)
print("Training ARIMA model...")
# best_model already fit on training data

print("Making predictions (rolling 1-step-ahead)...")
if steps > 0:
    history = train_series_model.copy()
    fc_model_space = []
    for i in range(steps):
        try:
            res_i = ARIMA(
                history,
                order=best_order,
                trend=best_trend,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(method_kwargs={"warn_convergence": False})
            fc = res_i.forecast(steps=1)[0]
        except Exception:
            # Fallback: use last observed value in model space
            fc = history[-1]
        fc_model_space.append(fc)
        next_obs = np.log(test_series[i] + _EPS) if USE_LOG else test_series[i]
        history = np.append(history, next_obs)
    fc_model_space = np.asarray(fc_model_space)
    if USE_LOG:
        forecast = np.exp(fc_model_space) - _EPS
    else:
        forecast = fc_model_space
else:
    forecast = np.array([])

# Scale predictions for metrics to align with other examples
predicted_test_scaled = _scaler.transform(forecast.reshape(-1, 1)).flatten() if steps > 0 else np.array([])

# Metrics on scaled values (consistent with other scripts)
if steps > 0:
    # Train metrics (align by length in case of small offsets)
    _n_train = min(len(train_scaled), len(predicted_train_scaled))
    train_mse = mean_squared_error(train_scaled[:_n_train], predicted_train_scaled[:_n_train])
    train_mae = mean_absolute_error(train_scaled[:_n_train], predicted_train_scaled[:_n_train])
    print(f"Train MSE: {train_mse:.6f}")
    print(f"Train MAE: {train_mae:.6f}")

    test_mse = mean_squared_error(test_scaled, predicted_test_scaled)
    test_mae = mean_absolute_error(test_scaled, predicted_test_scaled)
    print(f"Test MSE: {test_mse:.6f}")
    print(f"Test MAE: {test_mae:.6f}")
else:
    print("Insufficient test samples for evaluation.")

# Prepare arrays for plotting (matching other scripts)
seq_length = 0  # Not used for ARIMA, but keep plotting indices consistent
original_train = data_scaled[:train_length]
original_test = data_scaled[train_length:train_length + steps]

time_steps_train = np.arange(0, train_length)
time_steps_test = np.arange(train_length, train_length + steps)

# Plot predictions vs original data
plt.figure(figsize=(12, 6))
plt.plot(time_steps_train, original_train, label='Original Train Data')
# Align predicted train length to time_steps_train
predicted_train_plot = predicted_train_scaled[:len(time_steps_train)]
plt.plot(time_steps_train, predicted_train_plot, label='Predicted Train Data', linestyle='--')
plt.plot(time_steps_test, original_test, label='Original Test Data')
if steps > 0:
    plt.plot(time_steps_test, predicted_test_scaled, label='Predicted Test Data', linestyle='--')
plt.title('ARIMA Model Predictions vs. Original Data')
plt.xlabel('Time Step')
plt.ylabel('Scaled Value')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("ARIMA time series forecasting completed successfully!")
