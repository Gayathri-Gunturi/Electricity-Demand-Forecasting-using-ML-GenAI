import pandas as pd
import joblib

from energy_assistant import generate_energy_insight


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

probability_threshold = 0.20


# -----------------------------
# Load historical data
# -----------------------------

df = pd.read_csv("../data/load_data.csv")

df["timestamp"] = pd.to_datetime(df["timestamp"])

df = df.sort_values("timestamp").reset_index(drop=True)


# -----------------------------
# Get latest observation
# -----------------------------

latest_index = len(df) - 1

latest = df.iloc[latest_index]

latest_timestamp = latest["timestamp"]

next_timestamp = latest_timestamp + pd.Timedelta(hours=1)


# -----------------------------
# Create next-hour features
# -----------------------------

next_features = {
    "hour": next_timestamp.hour,
    "day_of_week": next_timestamp.dayofweek,

    # Recent observations
    "lag_1": df.iloc[latest_index]["load"],
    "lag_2": df.iloc[latest_index - 1]["load"],
    "lag_3": df.iloc[latest_index - 2]["load"],
    "lag_6": df.iloc[latest_index - 5]["load"],
    "lag_12": df.iloc[latest_index - 11]["load"],
    "lag_24": df.iloc[latest_index - 23]["load"],

    # Weekly lag
    "lag_168": df.iloc[latest_index - 167]["load"]
}


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


X_next = pd.DataFrame(
    [next_features],
    columns=features
)


# -----------------------------
# Forecast next-hour load
# -----------------------------

predicted_load = forecast_model.predict(X_next)[0]


# -----------------------------
# Calculate peak probability
# -----------------------------

peak_probability = peak_model.predict_proba(X_next)[0][1]

peak_warning = (
    peak_probability >= probability_threshold
)


# -----------------------------
# Generate AI energy insight
# -----------------------------

insight = generate_energy_insight(
    current_load=latest["load"],
    predicted_load=predicted_load,
    peak_probability=peak_probability,
    peak_threshold=peak_threshold
)


# -----------------------------
# Display forecast results
# -----------------------------

print("\nNext-Hour Energy Forecast")
print("---------------------------")

print(
    "Current time:",
    latest_timestamp
)

print(
    "Forecast time:",
    next_timestamp
)

print(
    "Current load:",
    latest["load"],
    "kW"
)

print(
    "Predicted load:",
    round(predicted_load, 2),
    "kW"
)

print(
    "Peak probability:",
    round(peak_probability * 100, 2),
    "%"
)

print(
    "Peak threshold:",
    peak_threshold,
    "kW"
)

print(
    "Peak warning:",
    "YES" if peak_warning else "NO"
)


# -----------------------------
# Display AI insight
# -----------------------------

print("\nAI Energy Assistant")
print("---------------------------")

print(insight)