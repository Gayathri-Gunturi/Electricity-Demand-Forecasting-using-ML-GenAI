Electricity Demand Forecasting using ML & GenAI

An end-to-end machine learning project for electricity demand forecasting, peak-demand detection, workload-shifting simulation, and GenAI-powered energy insights.

The system forecasts the next-hour electricity load, identifies potential peak-demand periods, simulates shifting flexible workloads away from peak periods, and uses an LLM to explain the results in natural language.

🚀 Features
Electricity Load Forecasting
Predicts next-hour electricity demand using historical load patterns.
Uses lag-based and time-based features.
Uses HistGradientBoostingRegressor.
Peak-Demand Detection
Learns a peak threshold from the training data.
Uses HistGradientBoostingClassifier to estimate peak probability.
Uses a probability threshold of 20% for peak warnings.
Workload-Shifting Simulation
Simulates shifting flexible electricity demand away from a predicted peak.
Calculates the resulting simulated load.
Determines whether the simulated load falls below the peak threshold.
GenAI Energy Assistant
Uses Groq's openai/gpt-oss-120b model.
Converts numerical forecasting results into concise natural-language insights.
Explains peak risk and simulated workload shifting.
Uses only the provided forecasting and simulation data.
🏗️ System Architecture
Historical Electricity Data
            │
            ▼
    Data Preparation
            │
            ▼
    Feature Engineering
            │
            ▼
    Load Forecasting Model
            │
            ├──────────────► Predicted Load
            │
            ▼
     Peak Classifier
            │
            ▼
      Peak Probability
            │
            ▼
   Workload-Shift Simulation
            │
            ▼
     Simulated Load
            │
            ▼
     GenAI Energy Assistant
            │
            ▼
 Natural-Language Energy Insight
📊 Dataset

This project uses the Electricity Load Diagrams 2011–2014 dataset from the UCI Machine Learning Repository.

The project uses an hourly aggregated version of the dataset containing:

321 electricity-consumption time series
Hourly observations
Data covering 2012–2014
Electricity consumption measured in kW

For the initial forecasting model, one representative time series (T1) is used.

⚙️ Feature Engineering

The forecasting model uses the following features:

Feature	Description
hour	Hour of the day
day_of_week	Day of the week
lag_1	Previous hour's load
lag_2	Load from 2 hours earlier
lag_3	Load from 3 hours earlier
lag_6	Load from 6 hours earlier
lag_12	Load from 12 hours earlier
lag_24	Load from the previous day
lag_168	Load from the previous week

A chronological 80/20 train-test split is used to preserve the temporal order of the data.

🤖 Machine Learning Models
Load Forecasting
HistGradientBoostingRegressor

Final test performance:

Metric	Value
MAE	3.67 kW
RMSE	7.57 kW

A naive previous-hour baseline achieved:

Metric	Baseline
MAE	4.18 kW
RMSE	9.80 kW

The final forecasting model therefore improves upon the naive baseline on both metrics.

Peak Detection

The peak threshold is calculated from the 95th percentile of the training load:

Peak threshold = 72 kW

The classifier estimates the probability that the upcoming load belongs to the peak-demand class.

Because peak observations are relatively rare, peak recall is considered alongside overall accuracy.

⚡ Workload-Shifting Simulation

The project also includes a simple what-if simulation for flexible electricity workloads.

For example:

Predicted load       = 82 kW
Peak threshold       = 72 kW
Flexible workload    = 15 kW

Without shifting:

82 kW → Peak

Simulated workload shifting:

82 - 15 = 67 kW

Result:

67 kW → Below peak threshold

This is a simulation only. It does not represent an actual reduction in electricity consumption.

🧠 GenAI Energy Assistant

The GenAI layer receives the forecasting and simulation results and generates a concise explanation.

Example:

Predicted load: 82 kW
Peak threshold: 72 kW
Peak probability: 78%
Flexible workload: 15 kW
Simulated load after shifting: 67 kW

The assistant can explain:

Expected demand
Peak-demand likelihood
Effect of simulated workload shifting
A practical recommendation

The prompt is designed to prevent the model from inventing electricity prices, savings, appliances, equipment, or external conditions.

📁 Project Structure
Smart-Energy-Forecasting/
│
├── data/
│   ├── electricity_hourly/
│   │   └── electricity_hourly_dataset.tsf
│   └── load_data.csv
│
├── models/
│   ├── load_forecast_model.pkl
│   ├── peak_classifier.pkl
│   └── peak_threshold.pkl
│
├── src/
│   ├── parse_tsf.py
│   ├── prepare_data.py
│   ├── create_features.py
│   ├── train_and_save_models.py
│   ├── forecast_next.py
│   ├── energy_assistant.py
│   └── workload_shift.py
│
├── notebooks/
│
├── app/
│
├── requirements.txt
├── .gitignore
└── README.md
🛠️ Tech Stack

Languages

Python

Machine Learning

Pandas
NumPy
Scikit-learn
Joblib

Generative AI

Groq API
openai/gpt-oss-120b

Data Visualization

Matplotlib
Seaborn

Development

Jupyter Notebook
VS Code
Git/GitHub
▶️ Running the Project
1. Clone the repository
git clone <your-repository-url>
cd Smart-Energy-Forecasting
2. Create a virtual environment
python -m venv venv

Activate it on Windows:

venv\Scripts\activate
3. Install dependencies
pip install -r requirements.txt
4. Set the Groq API key

Create a .env file or set the environment variable:

GROQ_API_KEY=your_api_key_here

Never commit the API key to GitHub.

5. Prepare the data

Run the data preparation scripts in order:

python src/parse_tsf.py
python src/prepare_data.py
python src/create_features.py
6. Train and save the models
python src/train_and_save_models.py

This generates:

models/
├── load_forecast_model.pkl
├── peak_classifier.pkl
└── peak_threshold.pkl
7. Run the forecasting pipeline
python src/forecast_next.py

The script produces:

Current load
Next-hour forecast
Peak probability
Peak warning
Peak threshold
AI-generated energy insight
8. Run the workload simulation
python src/workload_shift.py
🔐 Environment Variables

Create a local .env file:

GROQ_API_KEY=your_api_key_here

Add .env to .gitignore:

.env
venv/
__pycache__/
*.pyc
📌 Example Output
Next-Hour Energy Forecast
---------------------------
Current time: 2014-12-31 23:00:01
Forecast time: 2015-01-01 00:00:01
Current load: 11.0 kW
Predicted load: 11.32 kW
Peak probability: 0.04 %
Peak threshold: 72.0 kW
Peak warning: NO

AI Energy Assistant
---------------------------
Expected demand:
The load is projected to rise slightly from 11.00 kW
to 11.32 kW in the next hour.

Peak-demand assessment:
The upcoming hour is not a peak-demand period.

Recommendation:
No immediate peak-management action is indicated.
🎯 Project Goals

This project demonstrates an end-to-end workflow combining:

Time-series feature engineering
Machine learning forecasting
Imbalanced classification
Peak-demand analysis
What-if workload simulation
Generative AI
Model-to-LLM integration
⚠️ Limitations
The current implementation uses one representative electricity time series.
Weather, electricity pricing, holidays, and other external factors are not included.
Workload shifting is a simulated scenario rather than real-world control.
GenAI recommendations are grounded only in the numerical information supplied to the model.
The project is intended for experimentation and demonstration, not operational electricity-grid control.
📚 Dataset Source

Electricity Load Diagrams 2011–2014 — UCI Machine Learning Repository

Dataset DOI: 10.24432/C58C86

The hourly aggregated data used in this project is derived from the original electricity load dataset.

👩‍💻 Author

Gayathri Gunturi