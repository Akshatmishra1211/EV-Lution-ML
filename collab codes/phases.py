# ============================================
# PHASE 1 — Synthetic Dataset Generation (Final)
# ============================================

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import os

# -----------------------------
# AUTO-OVERWRITE OLD FILES
# -----------------------------
files_to_remove = [
    "ev_synthetic_dataset.csv",
    "ev_stations.csv"
]

for f in files_to_remove:
    if os.path.exists(f):
        os.remove(f)
        print(f"Removed old file: {f}")


# -----------------------------
# 1. Generate Station Metadata
# -----------------------------
def generate_stations(n=50, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    station_types = ["residential", "office", "mall", "highway"]
    records = []

    for i in range(n):
        st_id = f"S{i+1:04d}"
        st_type = random.choice(station_types)

        # base price ranges by station type
        type_price = {
            "residential": (6, 8),
            "office":      (7, 9),
            "mall":        (8, 12),
            "highway":     (10, 14)
        }

        base_price = random.uniform(*type_price[st_type])

        # Bangalore-like random GPS coordinates
        lat = 12.9 + np.random.randn() * 0.03
        lon = 77.5 + np.random.randn() * 0.03

        num_chargers = random.choice([2, 4, 6, 8, 10, 12])

        records.append({
            "station_id": st_id,
            "station_type": st_type,
            "lat": lat,
            "lon": lon,
            "num_chargers": num_chargers,
            "base_price": round(base_price, 2)
        })

    return pd.DataFrame(records)


stations = generate_stations()
print("Stations shape:", stations.shape)
print(stations.head())


# -----------------------------
# 2. Weather + Dynamic Pricing
# -----------------------------
def simulate_weather(ts):
    """Generate synthetic temperature + rain."""
    day = ts.timetuple().tm_yday
    temp = 25 + 5*np.sin(2*np.pi*day/365) + np.random.randn() * 1
    rain = np.random.choice([0,1], p=[0.92, 0.08])
    return round(temp, 2), rain


def dynamic_price(base_price, hour, station_type):
    """Add surge pricing for different station types."""
    surge = 0
    if station_type in ["office", "mall"] and 18 <= hour <= 22:
        surge = 0.20
    if station_type == "highway" and (hour <= 6 or hour >= 22):
        surge = 0.25
    if station_type == "residential" and 19 <= hour <= 23:
        surge = 0.15

    return round(base_price * (1 + surge), 2)


# -----------------------------
# 3. Base Occupancy Patterns
# -----------------------------
def base_occupancy_pattern(st_type, hour):
    """24-hour utilization pattern per station type."""
    patterns = {
        "residential": [0.1,0.1,0.08,0.05,0.05,0.1,0.2,0.3,0.35,0.45,0.5,0.45,
                        0.4,0.35,0.3,0.28,0.3,0.4,0.6,0.7,0.65,0.5,0.3,0.2],

        "office":      [0.05,0.05,0.05,0.05,0.1,0.25,0.4,0.6,0.75,0.85,0.9,0.85,
                        0.8,0.75,0.7,0.65,0.7,0.75,0.5,0.3,0.2,0.1,0.08,0.05],

        "mall":        [0.05,0.05,0.05,0.05,0.05,0.1,0.15,0.2,0.3,0.5,0.65,0.8,
                        0.85,0.9,0.95,0.9,0.85,0.8,0.75,0.6,0.4,0.25,0.1,0.05],

        "highway":     [0.2,0.25,0.3,0.35,0.4,0.5,0.45,0.4,0.35,0.3,0.25,0.2,
                        0.2,0.25,0.3,0.35,0.4,0.5,0.6,0.7,0.75,0.8,0.7,0.5],
    }
    return patterns[st_type][hour]


# -----------------------------
# 4. Generate Full Dataset (2 years)
# -----------------------------
def generate_full_dataset(stations_df, years=2):
    records = []
    start = datetime(2023,1,1)
    total_hours = years * 365 * 24   # 8760 hours per year

    for _, row in stations_df.iterrows():
        st_id = row["station_id"]
        st_type = row["station_type"]
        base_price = row["base_price"]
        num_c = row["num_chargers"]

        for h in range(total_hours):
            ts = start + timedelta(hours=h)
            hour = ts.hour
            dow = ts.weekday()

            temp, rain = simulate_weather(ts)
            dyn_price = dynamic_price(base_price, hour, st_type)

            base_occ = base_occupancy_pattern(st_type, hour)

            # adjust for weekend
            if st_type == "office" and dow >= 5:
                base_occ *= 0.35

            # adjust for rain
            if rain == 1:
                base_occ *= 0.8

            occ_prob = np.clip(base_occ + np.random.randn()*0.05, 0, 1)
            occupied = np.random.binomial(num_c, occ_prob)
            demand = occupied * np.random.uniform(8, 20)

            records.append({
                "timestamp": ts,
                "station_id": st_id,
                "station_type": st_type,
                "hour": hour,
                "day_of_week": dow,
                "temperature": temp,
                "rain": rain,
                "num_chargers": num_c,
                "occupied_count": occupied,
                "available_count": num_c - occupied,
                "utilization": round(occupied/num_c, 3),
                "dynamic_price": dyn_price,
                "demand_kwh": round(demand, 2)
            })

    return pd.DataFrame(records)


df = generate_full_dataset(stations)
print("\nGenerated dataset:", df.shape)
print(df.head())


# -----------------------------
# 5. SAVE Phase 1 output
# -----------------------------
df.to_csv("ev_synthetic_dataset.csv", index=False)
stations.to_csv("ev_stations.csv", index=False)

print("\nPhase 1 complete. Files saved:")
print(" - ev_synthetic_dataset.csv")
print(" - ev_stations.csv")


# ============================================
# PHASE 2 — Feature Engineering (Final Version)
# ============================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# -----------------------------
# LOAD Phase 1 output
# -----------------------------
df = pd.read_csv("ev_synthetic_dataset.csv", parse_dates=["timestamp"])
stations = pd.read_csv("ev_stations.csv")
print("Loaded df columns:", df.columns.tolist())

df.columns = df.columns.str.lower()
stations.columns = stations.columns.str.lower()

print("Loaded:", df.shape)

# -----------------------------
# SORT before lags
# -----------------------------
df = df.sort_values(["station_id", "timestamp"]).reset_index(drop=True)

# -----------------------------
# TIME FEATURES
# -----------------------------
df["month"]      = df["timestamp"].dt.month
df["day"]        = df["timestamp"].dt.day
df["is_weekend"] = (df["timestamp"].dt.weekday >= 5).astype(int)

# Cyclic encodings
df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24)
df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24)
df["dow_sin"]  = np.sin(2*np.pi*df["day_of_week"]/7)
df["dow_cos"]  = np.cos(2*np.pi*df["day_of_week"]/7)

# -----------------------------
# LAG FEATURES
# -----------------------------
lags = [1, 3, 6, 12, 24]

for lag in lags:
    df[f"occ_lag_{lag}"] = df.groupby("station_id")["utilization"].shift(lag)
    df[f"dmd_lag_{lag}"] = df.groupby("station_id")["demand_kwh"].shift(lag)

# -----------------------------
# ROLLING WINDOWS
# -----------------------------
windows = [3, 6, 24]

for w in windows:
    df[f"occ_roll_{w}"] = (df.groupby("station_id")["utilization"]
                              .rolling(w).mean().reset_index(level=0, drop=True))
    df[f"dmd_roll_{w}"] = (df.groupby("station_id")["demand_kwh"]
                              .rolling(w).mean().reset_index(level=0, drop=True))

# -----------------------------
# ONE-HOT ENCODE station_type
# -----------------------------
df = pd.get_dummies(df, columns=["station_type"], drop_first=True)

# -----------------------------
# DROP ONLY LAG/ROLL NaNs
# -----------------------------
lagroll_cols = [c for c in df.columns if "lag" in c or "roll" in c]
df = df.dropna(subset=lagroll_cols).reset_index(drop=True)

print("After features:", df.shape)

# -----------------------------
# TRAIN / VAL / TEST SPLIT (time-based)
# -----------------------------
timestamps = df["timestamp"].sort_values().unique()

N = len(timestamps)
train_end = int(N * 0.70)
val_end   = int(N * 0.85)

train_ts = timestamps[:train_end]
val_ts   = timestamps[train_end:val_end]
test_ts  = timestamps[val_end:]

train_df = df[df["timestamp"].isin(train_ts)]
val_df   = df[df["timestamp"].isin(val_ts)]
test_df  = df[df["timestamp"].isin(test_ts)]

print("Train:", train_df.shape)
print("Val  :", val_df.shape)
print("Test :", test_df.shape)

# -----------------------------
# SCALING NUMERIC FEATURES
# -----------------------------
num_features = [
    "temperature", "dynamic_price", "utilization", "demand_kwh"
] + [f"occ_lag_{l}" for l in lags] \
  + [f"dmd_lag_{l}" for l in lags] \
  + [f"occ_roll_{w}" for w in windows] \
  + [f"dmd_roll_{w}" for w in windows]

scaler = StandardScaler()
scaler.fit(train_df[num_features])

train_df[num_features] = scaler.transform(train_df[num_features])
val_df[num_features]   = scaler.transform(val_df[num_features])
test_df[num_features]  = scaler.transform(test_df[num_features])

joblib.dump(scaler, "scaler.pkl")

# -----------------------------
# SAVE PROCESSED FILES
# -----------------------------
train_df.to_csv("train_processed.csv", index=False)
val_df.to_csv("val_processed.csv", index=False)
test_df.to_csv("test_processed.csv", index=False)

print("\nPhase 2 complete — processed CSVs saved:")
print(" - train_processed.csv")
print(" - val_processed.csv")
print(" - test_processed.csv")
print(" - scaler.pkl")

# ================================================
# PHASE 4 — TCN Deep Learning (FINAL FIXED VERSION)
# ================================================

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from sklearn.preprocessing import LabelEncoder
import joblib
import time
import matplotlib.pyplot as plt

print("TF version:", tf.__version__)
AUTOTUNE = tf.data.AUTOTUNE

# -----------------------------
# LOAD Phase 2 OUTPUT FILES
# -----------------------------
train_df = pd.read_csv("train_processed.csv", parse_dates=["timestamp"])
val_df   = pd.read_csv("val_processed.csv", parse_dates=["timestamp"])
test_df  = pd.read_csv("test_processed.csv", parse_dates=["timestamp"])

scaler = joblib.load("scaler.pkl")

print("Loaded:", train_df.shape, val_df.shape, test_df.shape)

# -----------------------------
# DETECT WHICH COLUMNS WERE SCALED IN PHASE 2
# -----------------------------
scaled_cols = list(scaler.feature_names_in_)
print("Scaler numeric feature count:", len(scaled_cols))
print("Scaled cols:", scaled_cols)

# -----------------------------
# CREATE SEQUENCE FEATURE LIST
# -----------------------------
exclude_cols = ["timestamp"]
seq_feature_cols = [c for c in train_df.columns if c not in exclude_cols + ["station_id"]]

print("Total sequence features:", len(seq_feature_cols))

# -----------------------------
# LABEL ENCODE STATION ID
# -----------------------------
le_station = LabelEncoder()
train_df["station_id_enc"] = le_station.fit_transform(train_df["station_id"])
val_df["station_id_enc"]   = le_station.transform(val_df["station_id"])
test_df["station_id_enc"]  = le_station.transform(test_df["station_id"])

n_stations = len(le_station.classes_)
print("Number of stations:", n_stations)

# -----------------------------
# CREATE INPUT SEQUENCES (SLIDING WINDOWS)
# -----------------------------
def create_sequences(df, seq_cols, input_len=24, horizon=24):
    X, Xst, Y = [], [], []

    for sid, g in df.groupby("station_id_enc"):
        g = g.sort_values("timestamp").reset_index(drop=True)

        arr  = g[seq_cols].values.astype(np.float32)
        yarr = g["utilization"].values.astype(np.float32)

        L = len(g)
        max_start = L - input_len - horizon + 1

        for s in range(max_start):
            X.append(arr[s:s+input_len])
            Xst.append(sid)
            Y.append(yarr[s+input_len : s+input_len+horizon])

    return (
        np.array(X),
        np.array(Xst, dtype=np.int32),
        np.array(Y)
    )

INPUT_LEN = 72
HORIZON = 24

print("Generating sequences...")

X_train_seq, X_train_st, y_train = create_sequences(train_df, seq_feature_cols)
X_val_seq,   X_val_st,   y_val   = create_sequences(val_df,   seq_feature_cols)
X_test_seq,  X_test_st,  y_test  = create_sequences(test_df,  seq_feature_cols)

print("Train:", X_train_seq.shape, y_train.shape)
print("Val:  ", X_val_seq.shape)
print("Test: ", X_test_seq.shape)

# -----------------------------
# PARTIAL SCALING — ONLY SCALE THE NUMERIC FEATURES USED BY SCALER
# -----------------------------
def scale_sequences_partial(X_seq, scaler, seq_cols, scaled_cols):
    X_new = X_seq.copy()

    idx = [seq_cols.index(c) for c in scaled_cols if c in seq_cols]

    N, T, F = X_seq.shape
    subset = X_seq[:, :, idx].reshape(-1, len(idx))

    subset_scaled = scaler.transform(subset)
    X_new[:, :, idx] = subset_scaled.reshape(N, T, len(idx))

    return X_new

# NOTE: If retraining a clean model from scratch, do NOT run this block.
# Since the features were already scaled in Phase 2, running this scales them a second time.
# To keep this clean pipeline, we comment out the double-scaling:
#
# print("Scaling sequences...")
# X_train_seq = scale_sequences_partial(X_train_seq, scaler, seq_feature_cols, scaled_cols)
# X_val_seq   = scale_sequences_partial(X_val_seq,   scaler, seq_feature_cols, scaled_cols)
# X_test_seq  = scale_sequences_partial(X_test_seq,  scaler, seq_feature_cols, scaled_cols)
# print("Scaling complete.")

# -----------------------------
# BUILD TCN MODEL WITH STATION EMBEDDING
# -----------------------------
def tcn_block(x, filters, dilation, dropout=0.1):
    prev = x

    x = Conv1D(filters, 3, dilation_rate=dilation, padding="causal")(x)
    x = Activation("relu")(x)

    x = Conv1D(filters, 3, dilation_rate=dilation, padding="causal")(x)
    x = Activation("relu")(x)

    if prev.shape[-1] != filters:
        prev = Conv1D(filters, 1, padding="same")(prev)

    return Activation("relu")(Add()([prev, x]))

seq_input = Input(shape=(INPUT_LEN, len(seq_feature_cols)))
station_input = Input(shape=(), dtype="int32")

station_emb = Embedding(input_dim=n_stations, output_dim=8)(station_input)
station_emb_time = RepeatVector(INPUT_LEN)(station_emb)

x = Concatenate(axis=-1)([seq_input, station_emb_time])

x = tcn_block(x, 64, dilation=1)
x = tcn_block(x, 64, dilation=2)
x = tcn_block(x, 128, dilation=4)

x = Flatten()(x)
x = Dense(256, activation="relu")(x)
output = Dense(HORIZON)(x)

model = Model([seq_input, station_input], output)
model.compile(optimizer="adam", loss="mse", metrics=["mae"])
model.summary()

# -----------------------------
# DATASETS FOR TRAINING
# -----------------------------
BATCH_SIZE = 256

train_ds = tf.data.Dataset.from_tensor_slices(
    ((X_train_seq, X_train_st), y_train)
).shuffle(4096).batch(BATCH_SIZE).prefetch(AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices(
    ((X_val_seq,   X_val_st),   y_val)
).batch(BATCH_SIZE)

# -----------------------------
# TRAIN THE MODEL
# -----------------------------
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(patience=2)
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=callbacks
)

# -----------------------------
# SAVE MODEL + ENCODERS
# -----------------------------
model.save("tcn_model_final.keras")    # fixed!
joblib.dump(le_station, "station_label_encoder.pkl")
joblib.dump(seq_feature_cols, "seq_feature_cols.pkl")

print("Model + encoders saved successfully!")

# -----------------------------
# EVALUATE ON TEST SET
# -----------------------------
test_ds = tf.data.Dataset.from_tensor_slices(
    ((X_test_seq, X_test_st), y_test)
).batch(BATCH_SIZE)

pred_test = model.predict(test_ds)

from sklearn.metrics import mean_absolute_error, mean_squared_error

flat_mae = mean_absolute_error(y_test.flatten(), pred_test.flatten())
flat_rmse = np.sqrt(mean_squared_error(y_test.flatten(), pred_test.flatten()))

print("\nTEST MAE:", flat_mae)
print("TEST RMSE:", flat_rmse)

# -----------------------------
# PLOT PER-HOUR MAE
# -----------------------------
h_mae = np.mean(np.abs(y_test - pred_test), axis=0)

plt.figure(figsize=(8,3))
plt.plot(h_mae, marker="o")
plt.title("Per-hour MAE across 24-hour horizon")
plt.xlabel("Hour ahead")
plt.ylabel("MAE")
plt.grid()
plt.show()
