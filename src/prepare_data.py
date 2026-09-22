from pathlib import Path
import pandas as pd

# Load the parsed TSF data
file_path = Path("../data/electricity_hourly/electricity_hourly_dataset.tsf")

with open(file_path, "r") as file:
    lines = file.readlines()

data_started = False
series = []

for line in lines:
    line = line.strip()

    if line == "@data":
        data_started = True
        continue

    if data_started and line:
        parts = line.split(":")

        series_name = parts[0]
        timestamp = parts[1]
        values = parts[2]

        values = [float(x) for x in values.split(",")]

        series.append({
            "series_name": series_name,
            "start_timestamp": timestamp,
            "values": values
        })

df = pd.DataFrame(series)

# Select one electricity series
selected_series = df.iloc[0]

# Convert its values into a normal time series
values = selected_series["values"]

start_time = pd.to_datetime(
    selected_series["start_timestamp"],
    format="%Y-%m-%d %H-%M-%S"
)

timestamps = pd.date_range(
    start=start_time,
    periods=len(values),
    freq="h"
)

load_df = pd.DataFrame({
    "timestamp": timestamps,
    "load": values
})

print(load_df.head(10))

print("\nNumber of rows:", len(load_df))
print("\nStart:", load_df["timestamp"].min())
print("End:", load_df["timestamp"].max())
load_df.to_csv("../data/load_data.csv", index=False)