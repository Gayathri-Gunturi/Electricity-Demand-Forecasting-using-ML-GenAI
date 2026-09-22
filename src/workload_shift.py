import joblib


# Load the trained forecast model and peak threshold
forecast_model = joblib.load("../models/load_forecast_model.pkl")
peak_threshold = joblib.load("../models/peak_threshold.pkl")


# Example upcoming-hour conditions
predicted_load = 82.0
flexible_workload = 15.0


# Load before shifting
load_before = predicted_load


# Shift flexible workload away from the peak hour
load_after = max(0, predicted_load - flexible_workload)


print("\nWorkload-Shifting Simulation")
print("---------------------------")

print("Predicted load:", load_before, "kW")
print("Peak threshold:", peak_threshold, "kW")
print("Flexible workload:", flexible_workload, "kW")

print("\nBefore workload shifting:")
print("Load:", load_before, "kW")
print(
    "Peak status:",
    "PEAK" if load_before >= peak_threshold else "NORMAL"
)

print("\nAfter workload shifting:")
print("Load:", load_after, "kW")
print(
    "Peak status:",
    "PEAK" if load_after >= peak_threshold else "NORMAL"
)

reduction = load_before - load_after

print("\nLoad reduction:", reduction, "kW")