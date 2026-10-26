
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -----------------------------------
# REPRODUCIBILITY
# -----------------------------------

np.random.seed(42)
tf.random.set_seed(42)

# -----------------------------------
# 1. GENERATE DEMO SOLAR DATA
# -----------------------------------

hours = np.arange(24 * 60)

solar_power = 40 * np.maximum(
    0,
    np.sin(np.pi * ((hours % 24) - 6) / 12)
)

noise = np.random.normal(0, 1.5, len(hours))
solar_power = np.maximum(0, solar_power + noise)

# Generate timestamps matching load_model.py
timestamps = pd.date_range(
    start="2026-08-01 00:00:00",
    periods=len(hours),
    freq="h"
)

# Create historical solar dataset
data = pd.DataFrame({
    "Time": timestamps,
    "Hour": hours,
    "Solar Generation (MW)": solar_power
})

# Save synthetic historical solar data
data.to_csv("demo_solar_data.csv", index=False)

print("Historical solar dataset saved: demo_solar_data.csv")
print(f"Number of historical records: {len(data)}")

# -----------------------------------
# 2. NORMALIZE DATA
# -----------------------------------

scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(
    data[["Solar Generation (MW)"]]
)

# -----------------------------------
# 3. CREATE SEQUENCES
# Previous 24 hours -> next hour
# -----------------------------------

LOOKBACK = 24

X, y = [], []

for i in range(LOOKBACK, len(scaled_data)):
    X.append(scaled_data[i - LOOKBACK:i])
    y.append(scaled_data[i, 0])

X = np.array(X)
y = np.array(y)

# Chronological train/test split
split = int(len(X) * 0.8)

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"Training sequences: {len(X_train)}")
print(f"Testing sequences: {len(X_test)}")

# -----------------------------------
# 4. BUILD LSTM MODEL
# -----------------------------------

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(LOOKBACK, 1)),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(1)
])

model.compile(
    optimizer="adam",
    loss="mse"
)

# -----------------------------------
# 5. TRAIN MODEL
# -----------------------------------

model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.1,
    shuffle=False,
    verbose=1
)

# -----------------------------------
# 6. PREDICT TEST DATA
# -----------------------------------

predicted_scaled = model.predict(X_test, verbose=0)

actual = scaler.inverse_transform(
    y_test.reshape(-1, 1)
).flatten()

predicted = scaler.inverse_transform(
    predicted_scaled
).flatten()

# Solar generation cannot be negative
predicted = np.maximum(predicted, 0)

# -----------------------------------
# 7. EVALUATE MODEL
# -----------------------------------

mae = mean_absolute_error(actual, predicted)

rmse = np.sqrt(
    mean_squared_error(actual, predicted)
)

print("\n--- Solar LSTM Results ---")
print(f"Test MAE:  {mae:.3f} MW")
print(f"Test RMSE: {rmse:.3f} MW")

# -----------------------------------
# 8. SAVE MODEL AND TEST RESULTS
# -----------------------------------

model.save("solar_lstm.keras")

results = pd.DataFrame({
    "Actual Solar (MW)": actual,
    "Predicted Solar (MW)": predicted
})

results.to_csv(
    "solar_predictions.csv",
    index=False
)

print("\nModel saved: solar_lstm.keras")
print("Predictions saved: solar_predictions.csv")

# -----------------------------------
# 9. FORECAST NEXT 24 HOURS
# -----------------------------------

# Start with the latest 24 historical solar values
last_sequence = scaled_data[-LOOKBACK:].copy()

future_predictions = []

for _ in range(24):

    # Predict the next hour
    next_scaled = model.predict(
        last_sequence.reshape(1, LOOKBACK, 1),
        verbose=0
    )[0, 0]

    future_predictions.append(next_scaled)

    # Slide the sequence forward
    last_sequence = np.vstack([
        last_sequence[1:],
        [[next_scaled]]
    ])

# Convert predictions back to MW
future_solar = scaler.inverse_transform(
    np.array(future_predictions).reshape(-1, 1)
).flatten()

future_solar = np.maximum(future_solar, 0)

# Generate timestamps for the next 24 hours
future_timestamps = pd.date_range(
    start=timestamps[-1] + pd.Timedelta(hours=1),
    periods=24,
    freq="h"
)

# Save future solar forecast
future_df = pd.DataFrame({
    "Time": future_timestamps,
    "Forecast Hour": np.arange(1, 25),
    "Predicted Solar (MW)": future_solar
})

future_df.to_csv(
    "future_solar_forecast.csv",
    index=False
)

print("\nNext 24-Hour Solar Forecast:")
print(future_df.to_string(index=False))

print("\nSaved: future_solar_forecast.csv")

