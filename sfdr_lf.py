import numpy as np
import tensorflow as tf
from pathlib import Path
import pandas as pd
from tensorflow.keras import layers, Model

BASE_DIR = Path(__file__).resolve().parent

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ==========================================
# 1. MODEL CONFIGURATION
# ==========================================

NUM_REGIONS = 5
TIME_STEPS = 24
FORECAST_HORIZON = 24
EMBEDDING_DIM = 32
NUM_HOURS = 2000
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
BATCH_SIZE = 32
EPOCHS = 20

# ==========================================
# 2. GENERATE SYNTHETIC REGIONAL LOAD DATA
# ==========================================

def generate_synthetic_data(num_hours=NUM_HOURS):
    """Generate illustrative hourly loads for five synthetic regions."""
    time = np.arange(num_hours)
    regional_loads = []

    # A shared pattern creates some correlation between regions.
    shared_daily = 12 * np.sin(2 * np.pi * (time - 6) / 24)
    shared_weekly = 5 * np.sin(2 * np.pi * time / (24 * 7))

    for region in range(NUM_REGIONS):
        base_load = 70 + region * 15
        region_daily = (2 + region) * np.sin(
            2 * np.pi * (time - 7 - region) / 24
        )
        noise = np.random.normal(0, 2, num_hours)
        load = base_load + shared_daily + shared_weekly + region_daily + noise
        regional_loads.append(np.maximum(load, 0))

    # Shape: (hours, regions)
    return np.asarray(regional_loads, dtype=np.float32).T


load_data = generate_synthetic_data()

print("SFDRLF model")
print("------------")
print("Number of regions:", NUM_REGIONS)
print("Historical time steps:", TIME_STEPS)
print("Forecast horizon:", FORECAST_HORIZON)
print("Dataset shape:", load_data.shape)

# ==========================================
# 3. STATIC GRID TOPOLOGY
# ==========================================

# Illustrative regional connections only; not a real electrical grid.
A_static = np.array([
    [1, 1, 0, 0, 0],
    [1, 1, 1, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 1, 1, 1],
    [0, 0, 0, 1, 1]
], dtype=np.float32)


def normalize_adjacency(adjacency):
    """Symmetric degree normalization for an adjacency matrix with self-loops."""
    adjacency = np.asarray(adjacency, dtype=np.float32)
    degree = np.sum(adjacency, axis=-1)
    inv_sqrt_degree = np.power(np.maximum(degree, 1.0), -0.5)
    return inv_sqrt_degree[:, None] * adjacency * inv_sqrt_degree[None, :]


A_static_normalized = normalize_adjacency(A_static)
print("\nStatic adjacency matrix:")
print(A_static)

# ==========================================
# 4. CREATE CHRONOLOGICAL SUPERVISED SEQUENCES
# ==========================================

def create_sequences(data, time_steps=TIME_STEPS, horizon=FORECAST_HORIZON):
    """Map the previous 24 hours to the next 6 hours for every region."""
    X, y = [], []
    for end in range(time_steps, len(data) - horizon + 1):
        X.append(data[end - time_steps:end, :])
        y.append(data[end:end + horizon, :])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)


X_all, y_all = create_sequences(load_data)
print("Input sequence shape:", X_all.shape)
print("Target sequence shape:", y_all.shape)

# Split chronologically, not randomly.
n_samples = len(X_all)
train_end = int(n_samples * TRAIN_RATIO)
val_end = int(n_samples * (TRAIN_RATIO + VAL_RATIO))

X_train, y_train = X_all[:train_end], y_all[:train_end]
X_val, y_val = X_all[train_end:val_end], y_all[train_end:val_end]
X_test, y_test = X_all[val_end:], y_all[val_end:]

# Scale using training data only to avoid leakage.
train_max = float(np.max(X_train))
train_min = float(np.min(X_train))
scale_range = max(train_max - train_min, 1e-6)

def scale_load(values):
    return (values - train_min) / scale_range

def inverse_scale(values):
    return values * scale_range + train_min

X_train_s = scale_load(X_train)
X_val_s = scale_load(X_val)
X_test_s = scale_load(X_test)
y_train_s = scale_load(y_train)
y_val_s = scale_load(y_val)
y_test_s = scale_load(y_test)

# ==========================================
# 5. MODEL INPUTS
# ==========================================

load_input = layers.Input(
    shape=(TIME_STEPS, NUM_REGIONS),
    name="historical_load"
)

adjacency_input = layers.Input(
    shape=(NUM_REGIONS, NUM_REGIONS),
    name="static_adjacency"
)

# ==========================================
# 6. TEMPORAL BRANCH: REGION-WISE BiLSTM
# ==========================================

# Input shape: (batch, 24 hours, 5 regions)
# Rearrange to: (batch, 5 regions, 24 hours, 1 feature)

region_sequences = layers.Permute(
    (2, 1),
    name="regions_first"
)(load_input)

region_sequences = layers.Reshape(
    (NUM_REGIONS, TIME_STEPS, 1),
    name="region_sequence_input"
)(region_sequences)

# Apply a separate BiLSTM to each region's 24-hour sequence.
temporal_features = layers.TimeDistributed(
    layers.Bidirectional(
        layers.LSTM(EMBEDDING_DIM)
    ),
    name="regionwise_bilstm"
)(region_sequences)

# Project 64 BiLSTM features to the embedding dimension (32).
temporal_features = layers.Dense(
    EMBEDDING_DIM,
    activation="relu",
    name="temporal_projection"
)(temporal_features)

print("Temporal branch created: region-wise BiLSTM")
print("Temporal feature shape:", temporal_features.shape)
# ==========================================
# 7. SPATIAL BRANCH: BATCH-SAFE GCN
# ==========================================

# Mean load over the input window gives one feature per region.
spatial_features = layers.Lambda(
    lambda x: tf.reduce_mean(x, axis=1),
    name="regional_mean_load"
)(load_input)
spatial_features = layers.Reshape(
    (NUM_REGIONS, 1), name="spatial_feature_dimension"
)(spatial_features)

# Add trainable node features, then aggregate over the graph.
spatial_features = layers.Dense(
    EMBEDDING_DIM, activation="relu", name="spatial_input_projection"
)(spatial_features)

def batch_graph_convolution(inputs):
    node_features, adjacency = inputs
    degree = tf.reduce_sum(adjacency, axis=-1)
    inv_sqrt_degree = tf.pow(tf.maximum(degree, 1.0), -0.5)
    normalized = (
        inv_sqrt_degree[:, :, None]
        * adjacency
        * inv_sqrt_degree[:, None, :]
    )
    return tf.matmul(normalized, node_features)

spatial_features = layers.Lambda(
    batch_graph_convolution,
    name="static_graph_convolution"
)([spatial_features, adjacency_input])

spatial_features = layers.Dense(
    EMBEDDING_DIM, activation="relu", name="spatial_projection"
)(spatial_features)

# ==========================================
# 8. SPATIOTEMPORAL FEATURE FUSION
# ==========================================

combined_features = layers.Concatenate(
    axis=-1, name="temporal_spatial_concat"
)([temporal_features, spatial_features])

fusion_gate = layers.Dense(
    EMBEDDING_DIM * 2,
    activation="sigmoid",
    name="feature_fusion_gate"
)(combined_features)

fused_projection = layers.Dense(
    EMBEDDING_DIM * 2,
    activation="relu",
    name="feature_projection"
)(combined_features)

fused_features = layers.Multiply(
    name="gated_feature_fusion"
)([fusion_gate, fused_projection])

# ==========================================
# 9. DYNAMIC GRAPH LEARNING AND AGGREGATION
# ==========================================

source_features = layers.Dense(
    EMBEDDING_DIM, activation="tanh", name="dynamic_source_features"
)(fused_features)
target_features = layers.Dense(
    EMBEDDING_DIM, activation="tanh", name="dynamic_target_features"
)(fused_features)

def calculate_dynamic_graph(inputs):
    source, target = inputs
    scores = tf.matmul(source, target, transpose_b=True)
    scores = scores / tf.sqrt(tf.cast(EMBEDDING_DIM, tf.float32))
    return tf.nn.softmax(scores, axis=-1)

dynamic_adjacency = layers.Lambda(
    calculate_dynamic_graph, name="dynamic_adjacency"
)([source_features, target_features])

# Combine fixed prior connectivity and learned feature relationships.
def combine_and_normalize_graph(inputs):
    static_adj, dynamic_adj = inputs
    combined = 0.5 * static_adj + 0.5 * dynamic_adj
    degree = tf.reduce_sum(combined, axis=-1)
    inv_sqrt_degree = tf.pow(tf.maximum(degree, 1e-6), -0.5)
    return (
        inv_sqrt_degree[:, :, None]
        * combined
        * inv_sqrt_degree[:, None, :]
    )

combined_adjacency = layers.Lambda(
    combine_and_normalize_graph, name="combined_dynamic_static_graph"
)([adjacency_input, dynamic_adjacency])

dynamic_spatial_features = layers.Lambda(
    lambda inputs: tf.matmul(inputs[0], inputs[1]),
    name="dynamic_graph_aggregation"
)([combined_adjacency, fused_features])

# Residual connection retains each region's own fused information.
regional_features = layers.Add(name="spatial_residual")(
    [fused_features, dynamic_spatial_features]
)
regional_features = layers.Dense(
    EMBEDDING_DIM, activation="relu", name="forecast_hidden"
)(regional_features)

# ==========================================
# 10. SIX-HOUR FORECASTING HEAD
# ==========================================

# Each region gets six future values.
forecast_flat = layers.Dense(
    FORECAST_HORIZON, name="per_region_forecast"
)(regional_features)
forecast_output = layers.Permute(
    (2, 1), name="forecast_output"
)(forecast_flat)  # (batch, horizon, regions)

model = Model(
    inputs=[load_input, adjacency_input],
    outputs=forecast_output,
    name="SFDRLF"
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="mse",
    metrics=["mae"]
)

print("\nSFDRLF model summary:")
model.summary()

# ==========================================
# 11. TRAIN THE MODEL
# ==========================================

def repeat_adjacency(number_of_samples):
    return np.repeat(
        A_static_normalized[np.newaxis, :, :],
        number_of_samples,
        axis=0
    ).astype(np.float32)

A_train = repeat_adjacency(len(X_train_s))
A_val = repeat_adjacency(len(X_val_s))
A_test = repeat_adjacency(len(X_test_s))

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=4,
    restore_best_weights=True
)

history = model.fit(
    [X_train_s, A_train],
    y_train_s,
    validation_data=([X_val_s, A_val], y_val_s),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=False,
    callbacks=[early_stopping],
    verbose=1
)

# ==========================================
# 12. TEST AND EVALUATE
# ==========================================

test_loss, test_mae_scaled = model.evaluate(
    [X_test_s, A_test], y_test_s, verbose=0
)

predicted_scaled = model.predict([X_test_s, A_test], verbose=0)
predicted = inverse_scale(predicted_scaled)
actual = inverse_scale(y_test_s)

mae = float(np.mean(np.abs(actual - predicted)))
rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))

print("\nSFDRLF test results")
print("-------------------")
print(f"Test MAE:  {mae:.3f} MW")
print(f"Test RMSE: {rmse:.3f} MW")
print("Actual test output shape:", actual.shape)
print("Predicted output shape:", predicted.shape)

# Per-region MAE
for region in range(NUM_REGIONS):
    region_mae = np.mean(np.abs(actual[:, :, region] - predicted[:, :, region]))
    print(f"Region {region + 1} MAE: {region_mae:.3f} MW")

# ==========================================
# 13. SAVE MODEL AND PREDICTIONS
# ==========================================

model.save(BASE_DIR / "sfdr_lf_model.keras")

# Save a sample of test predictions with explicit horizon and region columns.
# Each row represents one input window, one forecast hour, and one region.
rows = []
for sample_idx in range(len(actual)):
    for horizon_idx in range(FORECAST_HORIZON):
        for region_idx in range(NUM_REGIONS):
            rows.append({
                "Test Window": sample_idx + 1,
                "Forecast Hour Ahead": horizon_idx + 1,
                "Region": f"Region {region_idx + 1}",
                "Actual Load (MW)": float(actual[sample_idx, horizon_idx, region_idx]),
                "Predicted Load (MW)": float(predicted[sample_idx, horizon_idx, region_idx])
            })

pd.DataFrame(rows).to_csv(BASE_DIR / "sfdr_lf_predictions.csv", index=False)

print("\nSaved model: sfdr_lf_model.keras")
print("Saved predictions: sfdr_lf_predictions.csv")
print("\nNOTE: This is a synthetic-data prototype with illustrative graph connections.")
