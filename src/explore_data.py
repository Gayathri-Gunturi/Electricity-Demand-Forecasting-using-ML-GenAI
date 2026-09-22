import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("../data/load_data.csv")

# Convert timestamp to datetime
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Create time-based features
df["hour"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.dayofweek

print(df.head())

print("\nAverage load by hour:")
print(df.groupby("hour")["load"].mean())

print("\nAverage load by day of week:")
print(df.groupby("day_of_week")["load"].mean())