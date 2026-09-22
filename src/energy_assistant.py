import os
from groq import Groq


# -----------------------------
# Get Groq API key
# -----------------------------

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY is not set.")


# -----------------------------
# Create Groq client
# -----------------------------

client = Groq(
    api_key=api_key
)


# -----------------------------
# Generate AI energy insight
# -----------------------------

def generate_energy_insight(
    current_load,
    predicted_load,
    peak_probability,
    peak_threshold,
    flexible_workload=0
):
    simulated_load = max(0, predicted_load - flexible_workload)

    prompt = f"""
You are an AI energy management assistant.

Analyze the following electricity demand forecast and workload-shifting scenario.

Current electricity load: {current_load:.2f} kW
Predicted next-hour load: {predicted_load:.2f} kW
Peak probability: {peak_probability * 100:.2f}%
Peak-demand threshold: {peak_threshold:.2f} kW

Flexible workload available for shifting: {flexible_workload:.2f} kW
Simulated load after shifting: {simulated_load:.2f} kW

Provide:

1. A short explanation of the expected demand.
2. Whether the upcoming hour appears to be a peak-demand period.
3. Explain the simulated effect of shifting the flexible workload.
4. Give a concise recommendation.

Rules:
- Use only the data provided above.
- Clearly describe workload shifting as a simulation.
- Do not claim that real electricity consumption was reduced.
- Do not assume specific appliances or equipment exist.
- Do not invent electricity prices, savings, or external conditions.
- Keep the response concise.
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": "You are a concise AI energy management assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,
        max_tokens=300
    )

    return response.choices[0].message.content