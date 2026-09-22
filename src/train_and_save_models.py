import pandas as pd
import joblib

from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    HistGradientBoostingClassifier
)

# Load data
df = pd.read_csv("../data/load_data.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

# -----------------------------
# Feature Engineering
# -----------------------------

df["hour"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.dayofweek

df["lag_1"] = df["load"].shift(1)
df["lag_2"] = df["load"].shift(2)
df["lag_3"] = df["load"].shift(3)
df["lag_6"] = df["load"].shift(6)
df["lag_12"] = df["load"].shift(12)
df["lag_24"] = df["load"].shift(24)
df["lag_168"] = df["load"].shift(168)

df = df.dropna()

# -----------------------------
# Train/Test Split
# -----------------------------

split_index = int(len(df) * 0.8)

train = df.iloc[:split_index].copy()
test = df.iloc[split_index:].copy()

features = [
    "hour",
    "day_of_week",
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_6",
    "lag_12",
    "lag_24",
    "lag_168"
]

X_train = train[features]
X_test = test[features]

# -----------------------------
# Load Forecasting Model
# -----------------------------

forecast_model = HistGradientBoostingRegressor(
    random_state=42
)

forecast_model.fit(
    X_train,
    train["load"]
)

# -----------------------------
# Peak Classification Model
# -----------------------------

peak_threshold = train["load"].quantile(0.95)

train["is_peak"] = (
    train["load"] >= peak_threshold
).astype(int)

peak_model = HistGradientBoostingClassifier(
    random_state=42
)

peak_model.fit(
    X_train,
    train["is_peak"]
)

# -----------------------------
# Save Models
# -----------------------------

joblib.dump(
    forecast_model,
    "../models/load_forecast_model.pkl"
)

joblib.dump(
    peak_model,
    "../models/peak_classifier.pkl"
)

# Save the peak threshold too
joblib.dump(
    peak_threshold,
    "../models/peak_threshold.pkl"
)

print("Models saved successfully!")
print("Peak threshold:", peak_threshold)