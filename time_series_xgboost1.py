# -*- coding: utf-8 -*-
"""time_series_xgboost4.py

XGBoost implementation for time series forecasting
Aligned with variable notation and structure from time_series_pytorch1.py
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import yfinance as yf
from xgboost import XGBRegressor
import xgboost

# Load Stock Market Data from Yahoo Finance

def load_stock_data():
    return yf.download('AAPL', start='2010-01-01', end='2022-02-26')


def get_raw_series():
    df = load_stock_data()
    time_series = df['Close'].values  # Use Close price as time series
    return time_series, df.index


# Function to create sequences (same as in PyTorch example)

def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data) - seq_length):
        x = data[i : (i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


# Check if GPU is available (for XGBoost)

gpu_available = xgboost.build_info().get("USE_CUDA", False)
# Select tree method based on GPU availability
tree_method = "gpu_hist" if gpu_available else "hist"
print(f"GPU available: {gpu_available}, Tree method: {tree_method}")

# Set random seed for reproducibility
np.random.seed(0)

# Sequence length and data preparation (mirrors PyTorch script)
seq_length = 50
# Load raw (unscaled) series and date index
data_raw, date_index = get_raw_series()
# Create sequences on raw values
X_raw, y_raw = create_sequences(data_raw, seq_length)
print()
print(f"created {len(X_raw)} samples")

# Chronological split (like PyTorch version)
train_length = 2800
X_train_raw, y_train_raw = X_raw[:train_length], y_raw[:train_length]
X_test_raw, y_test_raw = X_raw[train_length:], y_raw[train_length:]
print(f"train samples: {len(X_train_raw)}")
print(f"test samples: {len(X_test_raw)}")

# Fit scaler on TRAIN RANGE ONLY to avoid leakage
# Include all values that appear in training windows
values_for_scaler = data_raw[:train_length + seq_length]
_scaler = MinMaxScaler().fit(values_for_scaler.reshape(-1, 1))

# Helper functions to scale sequences and targets
scale_seq = lambda arr: _scaler.transform(arr.reshape(-1, 1)).reshape(arr.shape)
scale_target = lambda arr: _scaler.transform(arr.reshape(-1, 1)).flatten()

# Scale train/test sequences and targets
X_train = scale_seq(X_train_raw)
y_train = scale_target(y_train_raw)
X_test = scale_seq(X_test_raw)
y_test = scale_target(y_test_raw)

# Ensure explicit 2D (n_samples, seq_length) and 1D (n_samples,) shapes
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)
y_train = y_train.reshape(-1)
y_test = y_test.reshape(-1)

# Stationary target: predict delta to last value in window, then add back
y_train_delta = y_train - X_train[:, -1]
y_test_delta = y_test - X_test[:, -1]

# Initialize XGBoost model (stable, conservative defaults)
model = XGBRegressor(
    n_estimators=600,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_alpha=0.0,
    reg_lambda=1.0,
    random_state=0,
    tree_method=tree_method,
    verbosity=0,
)

# Ensure arrays are 2D float32 and contiguous for XGBoost
X_train = np.ascontiguousarray(X_train, dtype=np.float32)
X_test = np.ascontiguousarray(X_test, dtype=np.float32)
y_train_delta = np.asarray(y_train_delta, dtype=np.float32)
y_test_delta = np.asarray(y_test_delta, dtype=np.float32)
print("Array shapes:", X_train.shape, y_train_delta.shape, X_test.shape, y_test_delta.shape)

# Train the model on delta target
print("Training XGBoost model...")
model.fit(
    X_train,
    y_train_delta,
    eval_set=[(X_test, y_test_delta)],
    verbose=False,
)

# Predictions (predict delta and add back last value)
print("Making predictions...")
predicted_train_delta = model.predict(X_train)
predicted_test_delta = model.predict(X_test)
predicted_train = X_train[:, -1] + predicted_train_delta
predicted_test = X_test[:, -1] + predicted_test_delta

# Metrics
train_mse = mean_squared_error(y_train, predicted_train)
train_mae = mean_absolute_error(y_train, predicted_train)
test_mse = mean_squared_error(y_test, predicted_test)
test_mae = mean_absolute_error(y_test, predicted_test)

print(f"Train MSE: {train_mse:.6f}")
print(f"Train MAE: {train_mae:.6f}")
print(f"Test MSE: {test_mse:.6f}")
print(f"Test MAE: {test_mae:.6f}")

# Diagnostics for potential flattening in predictions
half = max(1, len(predicted_test) // 2)
std_first = float(np.std(predicted_test[:half]))
std_last = float(np.std(predicted_test[-half:]))
# Guard against NaNs in correlation when variance is 0
try:
    corr_first = float(np.corrcoef(y_test[:half], predicted_test[:half])[0, 1])
except Exception:
    corr_first = float('nan')
try:
    corr_last = float(np.corrcoef(y_test[-half:], predicted_test[-half:])[0, 1])
except Exception:
    corr_last = float('nan')
print(f"Test prediction std — first half: {std_first:.6f}, last half: {std_last:.6f}")
print(f"Test correlation    — first half: {corr_first:.3f}, last half: {corr_last:.3f}")
print("Last 10 predictions:", np.array2string(predicted_test[-10:], precision=6))
print("Last 10 actuals:    ", np.array2string(y_test[-10:], precision=6))

# Prepare arrays for plotting (matching PyTorch plot structure)
# Build a scaled full series (using train-fitted scaler) for plotting
data_scaled = _scaler.transform(data_raw.reshape(-1, 1)).flatten()
original_train = data_scaled[seq_length : train_length + seq_length]
time_steps_train = np.arange(seq_length, train_length + seq_length)

original_test = data_scaled[train_length + seq_length : train_length + seq_length + len(X_test)]
time_steps_test = np.arange(train_length + seq_length, train_length + seq_length + len(X_test))

# Plot predictions vs original data
plt.figure(figsize=(12, 6))
plt.plot(time_steps_train, original_train, label='Original Train Data')
plt.plot(time_steps_train, predicted_train, label='Predicted Train Data', linestyle='--')
plt.plot(time_steps_test, original_test, label='Original Test Data')
plt.plot(time_steps_test, predicted_test, label='Predicted Test Data', linestyle='--')
plt.title('XGBoost Model Predictions vs. Original Data')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("XGBoost time series forecasting completed successfully!")
