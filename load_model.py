import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ============================================================
# 0. REPRODUCIBILITY
# ============================================================
np.random.seed(42)
tf.random.set_seed(42)

# ============================================================
# 1. GENERATE SYNTHETIC HOURLY LOAD DATA
# ============================================================
N_HOURS = 24 * 60
LOOKBACK = 24
TRAIN_RATIO = 0.80

hours = np.arange(N_HOURS)

daily_pattern = (
    80
    + 12 * np.sin(2 * np.pi * (hours - 7) / 24)
    + 8 * np.sin(4 * np.pi * (hours - 18) / 24)
)
weekly_pattern = 5 * np.sin(2 * np.pi * hours / (24 * 7))
noise = np.random.normal(0, 2, len(hours))

load = np.maximum(daily_pattern + weekly_pattern + noise, 10)

timestamps = pd.date_range(
    start="2026-08-01 00:00:00",
    periods=N_HOURS,
    freq="h"
)

df = pd.DataFrame({
    "Time": timestamps,
    "Hour": hours,
    "Electricity Load (MW)": load
})
df.to_csv("demo_load_data.csv", index=False)

print("Historical load dataset saved: demo_load_data.csv")
print(f"Historical records: {len(df)}")

# ============================================================
# 2. CREATE TIME FEATURES
# These describe repeating hour-of-day and day-of-week cycles.
# ============================================================
hour_of_day = df["Time"].dt.hour.to_numpy()
day_of_week = df["Time"].dt.dayofweek.to_numpy()

hour_sin = np.sin(2 * np.pi * hour_of_day / 24)
hour_cos = np.cos(2 * np.pi * hour_of_day / 24)
week_sin = np.sin(2 * np.pi * day_of_week / 7)
week_cos = np.cos(2 * np.pi * day_of_week / 7)

time_features = np.column_stack([
    hour_sin, hour_cos, week_sin, week_cos
]).astype(np.float32)

# ============================================================
# 3. CHRONOLOGICAL SPLIT AND SCALING
# Fit the load scaler using training-period values only.
# ============================================================
split_time = int(N_HOURS * TRAIN_RATIO)

scaler = MinMaxScaler()
scaler.fit(df.loc[:split_time - 1, ["Electricity Load (MW)"]])

scaled_load = scaler.transform(
    df[["Electricity Load (MW)"]]
).astype(np.float32)

# Features per hour: scaled load + 4 time features.
all_features = np.column_stack([
    scaled_load[:, 0],
    time_features
]).astype(np.float32)

# ============================================================
# 4. CREATE SEQUENCES
# Previous 24 hours -> next hour's load.
# ============================================================
X, y, target_indices = [], [], []

for target_idx in range(LOOKBACK, N_HOURS):
    X.append(all_features[target_idx - LOOKBACK:target_idx])
    y.append(scaled_load[target_idx, 0])
    target_indices.append(target_idx)

X = np.asarray(X, dtype=np.float32)
y = np.asarray(y, dtype=np.float32)
target_indices = np.asarray(target_indices)

train_mask = target_indices < split_time
test_mask = target_indices >= split_time

X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[test_mask], y[test_mask]

print(f"Training sequences: {len(X_train)}")
print(f"Testing sequences: {len(X_test)}")

# ============================================================
# 5. BUILD MODEL
# ============================================================
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(LOOKBACK, X.shape[2])),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(1)
])

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

# ============================================================
# 6. TRAIN WITH EARLY STOPPING
# ============================================================
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.1,
    shuffle=False,
    callbacks=[early_stopping],
    verbose=1
)

best_epoch = int(np.argmin(history.history["val_loss"]) + 1)
print(f"\nBest validation epoch: {best_epoch}")

# ============================================================
# 7. PREDICT AND EVALUATE HELD-OUT TEST DATA
# ============================================================
predicted_scaled = model.predict(X_test, verbose=0)

actual = scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
predicted = scaler.inverse_transform(predicted_scaled).ravel()

mae = mean_absolute_error(actual, predicted)
rmse = np.sqrt(mean_squared_error(actual, predicted))

# Simple persistence baseline: next hour = most recent observed load.
last_input_load_scaled = X_test[:, -1, 0].reshape(-1, 1)
baseline = scaler.inverse_transform(last_input_load_scaled).ravel()
baseline_mae = mean_absolute_error(actual, baseline)
baseline_rmse = np.sqrt(mean_squared_error(actual, baseline))

print("\n--- Electricity Load LSTM Results ---")
print(f"Test MAE:             {mae:.3f} MW")
print(f"Test RMSE:            {rmse:.3f} MW")
print(f"Persistence MAE:      {baseline_mae:.3f} MW")
print(f"Persistence RMSE:     {baseline_rmse:.3f} MW")

# Save test results. Keep the columns expected by app.py.
test_results = pd.DataFrame({
    "Actual Load (MW)": actual,
    "Predicted Load (MW)": predicted
})
test_results.to_csv("load_predictions.csv", index=False)

model.save("load_lstm.keras")
print("\nModel saved: load_lstm.keras")
print("Test predictions saved: load_predictions.csv")

# ============================================================
# 8. FORECAST THE NEXT 24 HOURS
# Use recursive predictions, updating the load input each hour.
# Time features for future timestamps are known in advance.
# ============================================================
future_timestamps = pd.date_range(
    start=timestamps[-1] + pd.Timedelta(hours=1),
    periods=24,
    freq="h"
)

future_predictions_scaled = []

# Last 24 rows of model input features.
sequence = all_features[-LOOKBACK:].copy()

for future_time in future_timestamps:
    next_scaled = float(model.predict(
        sequence.reshape(1, LOOKBACK, sequence.shape[1]),
        verbose=0
    )[0, 0])

    future_predictions_scaled.append(next_scaled)

    future_hour = future_time.hour
    future_dow = future_time.dayofweek

    next_features = np.array([
        next_scaled,
        np.sin(2 * np.pi * future_hour / 24),
        np.cos(2 * np.pi * future_hour / 24),
        np.sin(2 * np.pi * future_dow / 7),
        np.cos(2 * np.pi * future_dow / 7)
    ], dtype=np.float32)

    sequence = np.vstack([sequence[1:], next_features])

future_load = scaler.inverse_transform(
    np.asarray(future_predictions_scaled).reshape(-1, 1)
).ravel()
future_load = np.maximum(future_load, 0)

future_df = pd.DataFrame({
    "Time": future_timestamps,
    "Forecast Hour": np.arange(1, 25),
    "Predicted Load (MW)": future_load
})
future_df.to_csv("future_load_forecast.csv", index=False)

print("\nNext 24-hour load forecast:")
print(future_df.to_string(index=False))
print("\nSaved: future_load_forecast.csv")
print("\nNote: This is a synthetic demonstration dataset, not real grid data.")
