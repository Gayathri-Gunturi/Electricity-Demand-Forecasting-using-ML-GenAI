import pandas as pd
import joblib

# -----------------------------
# Load trained models
# -----------------------------

forecast_model = joblib.load(
    "../models/load_forecast_model.pkl"
)

peak_model = joblib.load(
    "../models/peak_classifier.pkl"
)

peak_threshold = joblib.load(
    "../models/peak_threshold.pkl"
)

# Probability threshold for peak warning
probability_threshold = 0.20


# -----------------------------
# Load recent data
# -----------------------------

df = pd.read_csv("../data/load_data.csv")

df["timestamp"] = pd.to_datetime(df["timestamp"])

# -----------------------------
# Create features
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
# Select latest observation
# -----------------------------

latest = df.iloc[-1]

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

X = pd.DataFrame(
    [latest[features].values],
    columns=features
)

# -----------------------------
# Load Forecast
# -----------------------------

predicted_load = forecast_model.predict(X)[0]

# -----------------------------
# Peak Probability
# -----------------------------

peak_probability = peak_model.predict_proba(X)[0][1]

is_peak = peak_probability >= probability_threshold

# -----------------------------
# Display Results
# -----------------------------

print("\nEnergy Forecast")
print("---------------------------")

print("Timestamp:", latest["timestamp"])
print("Current Load:", latest["load"], "kW")
print("Predicted Load:", round(predicted_load, 2), "kW")

print(
    "Peak Probability:",
    round(peak_probability * 100, 2),
    "%"
)

print(
    "Peak Threshold:",
    peak_threshold,
    "kW"
)

print(
    "Peak Warning:",
    "YES" if is_peak else "NO"
)