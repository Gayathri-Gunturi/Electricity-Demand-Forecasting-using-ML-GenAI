import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv("../data/load_data.csv")

df["timestamp"] = pd.to_datetime(df["timestamp"])


# ============================================================
# 2. CREATE TIME FEATURES
# ============================================================

df["hour"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.dayofweek


# ============================================================
# 3. CREATE LAG FEATURES
# ============================================================

# Previous hour
df["lag_1"] = df["load"].shift(1)

# Same hour on previous day
df["lag_24"] = df["load"].shift(24)

# Same hour on previous week
df["lag_168"] = df["load"].shift(168)
df["lag_2"] = df["load"].shift(2)
df["lag_3"] = df["load"].shift(3)
df["lag_6"] = df["load"].shift(6)
df["lag_12"] = df["load"].shift(12)


# ============================================================
# 4. REMOVE MISSING VALUES
# ============================================================

df = df.dropna()


# ============================================================
# 5. SELECT FEATURES
# ============================================================
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

# ============================================================
# 6. TIME-BASED TRAIN / TEST SPLIT
# ============================================================

split_index = int(len(df) * 0.8)

train = df.iloc[:split_index]
test = df.iloc[split_index:]


X_train = train[features]
y_train = train["load"]

X_test = test[features]
y_test = test["load"]


print("Training data:")
print("Start:", train["timestamp"].min())
print("End:", train["timestamp"].max())
print("Rows:", len(train))

print("\nTesting data:")
print("Start:", test["timestamp"].min())
print("End:", test["timestamp"].max())
print("Rows:", len(test))


# ============================================================
# 7. CREATE MODEL
# ============================================================

model = HistGradientBoostingRegressor(
    random_state=42
)


# ============================================================
# 8. TRAIN MODEL
# ============================================================

model.fit(X_train, y_train)


# ============================================================
# 9. MAKE PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# 10. EVALUATE MODEL
# ============================================================

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)


print("\nML Model Results")
print("----------------")
print("MAE:", mae)
print("RMSE:", rmse)


# ============================================================
# 11. CREATE ERROR DATAFRAME
# ============================================================

errors = y_test.values - y_pred

error_df = test[["timestamp", "load"]].copy()

error_df["predicted"] = y_pred

error_df["error"] = errors

error_df["absolute_error"] = np.abs(errors)


# ============================================================
# 12. FIND LARGEST PREDICTION ERRORS
# ============================================================

print("\nLargest Prediction Errors")
print("-------------------------")

largest_errors = (
    error_df
    .sort_values(
        "absolute_error",
        ascending=False
    )
    .head(10)
)

print(largest_errors)


# ============================================================
# 13. AVERAGE ERROR BY HOUR
# ============================================================

error_df["hour"] = error_df["timestamp"].dt.hour

hourly_error = (
    error_df
    .groupby("hour")["absolute_error"]
    .mean()
    .round(2)
)

print("\nAverage Absolute Error by Hour")
print("------------------------------")
print(hourly_error)


# ============================================================
# 14. ACTUAL VS PREDICTED GRAPH
# ============================================================

plt.figure(figsize=(12, 5))

plt.plot(
    y_test.iloc[:168].values,
    label="Actual"
)

plt.plot(
    y_pred[:168],
    label="Predicted"
)

plt.xlabel("Hour")
plt.ylabel("Load (kW)")

plt.title(
    "Actual vs Predicted Electricity Load"
)

plt.legend()

plt.tight_layout()

plt.show()

# ============================================================
# 15. ANALYZE HIGH-DEMAND PERIODS
# ============================================================

peak_threshold = train["load"].quantile(0.95)

peak_data = error_df[
    error_df["load"] >= peak_threshold
]

print("\nPeak Load Analysis")
print("------------------")

print("95th percentile load:", peak_threshold)

print("Number of peak observations:", len(peak_data))

print(
    "Average actual load during peaks:",
    round(peak_data["load"].mean(), 2)
)

print(
    "Average predicted load during peaks:",
    round(peak_data["predicted"].mean(), 2)
)

print(
    "Average absolute error during peaks:",
    round(peak_data["absolute_error"].mean(), 2)
)

# ============================================================
# 16. ANALYZE PEAK LOAD BY HOUR
# ============================================================

peak_by_hour = (
    peak_data
    .groupby("hour")["load"]
    .agg(["count", "mean", "max"])
    .round(2)
)

print("\nPeak Load by Hour")
print("-----------------")
print(peak_by_hour)


# ============================================================
# 17. ANALYZE PEAK LOAD BY DAY OF WEEK
# ============================================================

peak_data["day_of_week"] = (
    peak_data["timestamp"].dt.dayofweek
)

peak_by_day = (
    peak_data
    .groupby("day_of_week")["load"]
    .agg(["count", "mean", "max"])
    .round(2)
)

print("\nPeak Load by Day of Week")
print("------------------------")
print(peak_by_day)