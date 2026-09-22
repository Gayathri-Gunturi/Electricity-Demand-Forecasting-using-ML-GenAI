import pandas as pd

# Load data
df = pd.read_csv("../data/load_data.csv")

df["timestamp"] = pd.to_datetime(df["timestamp"])

# Time-based features
df["hour"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.dayofweek

# Lag features
df["lag_1"] = df["load"].shift(1)
df["lag_24"] = df["load"].shift(24)
df["lag_168"] = df["load"].shift(168)

print(df.head(10))

print("\nAfter 168 hours:")
print(df.iloc[168:175])

# Remove rows without enough history
df = df.dropna()

features = [
    "hour",
    "day_of_week",
    "lag_1",
    "lag_24",
    "lag_168"
]

X = df[features]
y = df["load"]

print("\nFeatures:")
print(X.head())

print("\nTarget:")
print(y.head())

print("\nX shape:", X.shape)
print("y shape:", y.shape)

# Time-based train/test split
split_index = int(len(df) * 0.8)

train = df.iloc[:split_index]
test = df.iloc[split_index:]

X_train = train[features]
y_train = train["load"]

X_test = test[features]
y_test = test["load"]

print("\nTraining data:")
print("Start:", train["timestamp"].min())
print("End:", train["timestamp"].max())
print("Rows:", len(train))

print("\nTesting data:")
print("Start:", test["timestamp"].min())
print("End:", test["timestamp"].max())
print("Rows:", len(test))