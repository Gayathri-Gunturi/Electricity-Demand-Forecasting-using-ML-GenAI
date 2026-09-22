from energy_assistant import generate_energy_insight

flexible_workload = 15.0

insight = generate_energy_insight(
    current_load=68.0,
    predicted_load=82.0,
    peak_probability=0.78,
    peak_threshold=72.0,
    flexible_workload=flexible_workload
)

print("\nWorkload-Shifting AI Test")
print("---------------------------")
print("Current load: 68.0 kW")
print("Predicted load: 82.0 kW")
print("Peak probability: 78.0%")
print("Peak threshold: 72.0 kW")
print("Flexible workload: 15.0 kW")
print("Simulated load after shifting: 67.0 kW")

print("\nAI Energy Assistant")
print("---------------------------")
print(insight)