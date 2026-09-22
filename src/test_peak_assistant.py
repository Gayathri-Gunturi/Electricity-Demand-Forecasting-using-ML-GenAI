from energy_assistant import generate_energy_insight

current_load = 68.0
predicted_load = 82.0
peak_probability = 0.78
peak_threshold = 72.0

insight = generate_energy_insight(
    current_load=current_load,
    predicted_load=predicted_load,
    peak_probability=peak_probability,
    peak_threshold=peak_threshold
)

print("\nPeak Scenario Test")
print("---------------------------")
print("Current load:", current_load, "kW")
print("Predicted load:", predicted_load, "kW")
print("Peak probability:", peak_probability * 100, "%")
print("Peak threshold:", peak_threshold, "kW")

print("\nAI Energy Assistant")
print("---------------------------")
print(insight)