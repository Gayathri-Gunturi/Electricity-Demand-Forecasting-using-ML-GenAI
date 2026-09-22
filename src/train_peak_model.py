import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)


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

df["lag_1"] = df["load"].shift(1)
df["lag_2"] = df["load"].shift(2)
df["lag_3"] = df["load"].shift(3)
df["lag_6"] = df["load"].shift(6)
df["lag_12"] = df["load"].shift(12)
df["lag_24"] = df["load"].shift(24)
df["lag_168"] = df["load"].shift(168)


# ============================================================
# 4. REMOVE MISSING VALUES
# ============================================================

df = df.dropna()


# ============================================================
# 5. TIME-BASED TRAIN / TEST SPLIT
# ============================================================

split_index = int(len(df) * 0.8)

train = df.iloc[:split_index].copy()
test = df.iloc[split_index:].copy()


# ============================================================
# 6. DEFINE PEAK THRESHOLD
# ============================================================

peak_threshold = train["load"].quantile(0.95)

print("Peak threshold:", peak_threshold)


# ============================================================
# 7. CREATE PEAK LABEL
# ============================================================

train["is_peak"] = (
    train["load"] >= peak_threshold
).astype(int)

test["is_peak"] = (
    test["load"] >= peak_threshold
).astype(int)


print("\nTraining peak distribution")
print(train["is_peak"].value_counts())

print("\nTesting peak distribution")
print(test["is_peak"].value_counts())


# ============================================================
# 8. SELECT FEATURES
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


X_train = train[features]
y_train = train["is_peak"]

X_test = test[features]
y_test = test["is_peak"]


# ============================================================
# 9. CREATE CLASSIFIER
# ============================================================

model = HistGradientBoostingClassifier(
    random_state=42
)


# ============================================================
# 10. TRAIN
# ============================================================

model.fit(X_train, y_train)


# ============================================================
# 11. PREDICT
# ============================================================

y_prob = model.predict_proba(X_test)[:, 1]

threshold = 0.20

y_pred = (y_prob >= threshold).astype(int)


# ============================================================
# 12. EVALUATE
# ============================================================

print("\nPeak Classification Results")
print("---------------------------")

print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix")
print("----------------")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report")
print("---------------------")
print(classification_report(y_test, y_pred))