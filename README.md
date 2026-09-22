# ⚡ Smart Energy Forecasting

An end-to-end **Machine Learning + Generative AI** system for electricity demand forecasting, peak-demand detection, workload-shifting simulation, and AI-powered energy insights.

The system forecasts the **next-hour electricity demand**, estimates **peak-demand probability**, simulates shifting flexible workloads away from peak periods, and uses an LLM to convert the numerical results into concise natural-language insights.

---

## 🚀 Features

### 📈 Electricity Load Forecasting

* Predicts next-hour electricity demand using historical load patterns.
* Uses lag-based and time-based features.
* Uses `HistGradientBoostingRegressor`.
* Evaluated using MAE and RMSE.
* Compared against a naive previous-hour baseline.

### 🔴 Peak-Demand Detection

* Learns the peak-demand threshold from training data.
* Uses the **95th percentile** of training load to define peak demand.
* Uses `HistGradientBoostingClassifier` to estimate peak probability.
* Uses a probability threshold to generate peak warnings.
* Considers peak recall because peak observations are relatively rare.

### ⚡ Workload-Shifting Simulation

Simulates a **what-if scenario** where flexible electricity demand is shifted away from a predicted peak period.

Example:

```text
Predicted load       = 82 kW
Peak threshold       = 72 kW
Flexible workload    = 15 kW

Without shifting:
82 kW → Peak

After simulated shifting:
82 - 15 = 67 kW

67 kW → Below peak threshold
```

> **Note:** Workload shifting is a simulation only. It does not represent an actual reduction in electricity consumption.

### 🧠 GenAI Energy Assistant

Uses **Groq's `openai/gpt-oss-120b`** model to explain forecasting and simulation results.

The assistant can describe:

* Expected electricity demand
* Peak-demand likelihood
* Effect of simulated workload shifting
* A practical recommendation based only on the supplied results

The prompt is designed to prevent the LLM from inventing information such as electricity prices, appliances, savings, weather conditions, or equipment.

---

## 🏗️ System Architecture

```text
                 Historical Electricity Data
                           │
                           ▼
                    Data Preparation
                           │
                           ▼
                   Feature Engineering
                           │
                           ▼
                ┌──────────────────────┐
                │  Load Forecast Model │
                └──────────────────────┘
                           │
                           ▼
                    Predicted Load
                           │
                           ▼
                ┌──────────────────────┐
                │  Peak Classifier     │
                └──────────────────────┘
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
                ┌──────────────────────┐
                │ GenAI Energy         │
                │ Assistant            │
                └──────────────────────┘
                           │
                           ▼
                 Natural-Language Insight
```

---

## 📊 Dataset

This project uses the **Electricity Load Diagrams 2011–2014** dataset from the UCI Machine Learning Repository.

The processed dataset contains:

* **321 electricity-consumption time series**
* Hourly observations
* Data covering 2012–2014
* Electricity consumption measured in kW

For the initial forecasting model, one representative time series (**T1**) is used.

**Dataset DOI:** `10.24432/C58C86`

---

## ⚙️ Feature Engineering

The forecasting model uses historical load and temporal features.

| Feature       | Description                 |
| ------------- | --------------------------- |
| `hour`        | Hour of the day             |
| `day_of_week` | Day of the week             |
| `lag_1`       | Previous hour's load        |
| `lag_2`       | Load from 2 hours earlier   |
| `lag_3`       | Load from 3 hours earlier   |
| `lag_6`       | Load from 6 hours earlier   |
| `lag_12`      | Load from 12 hours earlier  |
| `lag_24`      | Load from the previous day  |
| `lag_168`     | Load from the previous week |

A chronological **80/20 train-test split** is used to preserve the temporal ordering of the data.

---

## 🤖 Machine Learning Models

### Load Forecasting

**Model:** `HistGradientBoostingRegressor`

#### Test Performance

| Metric |       Model | Naive Baseline |
| ------ | ----------: | -------------: |
| MAE    | **3.67 kW** |        4.18 kW |
| RMSE   | **7.57 kW** |        9.80 kW |

The machine learning model improves upon the previous-hour naive baseline on both evaluation metrics.

### Peak Detection

The peak threshold is calculated using the **95th percentile of the training load**.

```text
Peak threshold = 72 kW
```

The classifier estimates the probability that the upcoming load belongs to the peak-demand class.

Because peak observations are relatively rare, **peak recall is considered alongside overall classification performance**.

---

## 🧠 GenAI Pipeline

The GenAI layer receives structured numerical results from the forecasting pipeline.

### Input

```text
Predicted load: 82 kW
Peak threshold: 72 kW
Peak probability: 78%
Flexible workload: 15 kW
Simulated load: 67 kW
```

### Output

The LLM converts these values into a concise energy insight covering:

```text
Expected demand
        ↓
Peak-demand assessment
        ↓
Effect of workload shifting
        ↓
Practical recommendation
```

The LLM is **not used to perform the numerical forecasting**. It acts as an explanation layer on top of the machine learning pipeline.

---

## 📁 Project Structure

```text
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
```

---

## 🛠️ Tech Stack

**Languages**

* Python

**Machine Learning**

* Pandas
* NumPy
* Scikit-learn
* Joblib

**Generative AI**

* Groq API
* `openai/gpt-oss-120b`

**Data Visualization**

* Matplotlib
* Seaborn

**Development**

* Jupyter Notebook
* VS Code
* Git
* GitHub

---

## ▶️ Getting Started

### 1. Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd Smart-Energy-Forecasting
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the Groq API Key

Create a `.env` file:

```env
GROQ_API_KEY=your_api_key_here
```

Never commit your API key to GitHub.

### 5. Prepare the Dataset

Run the data preparation scripts:

```bash
python src/parse_tsf.py
python src/prepare_data.py
python src/create_features.py
```

### 6. Train the Models

```bash
python src/train_and_save_models.py
```

This generates:

```text
models/
├── load_forecast_model.pkl
├── peak_classifier.pkl
└── peak_threshold.pkl
```

### 7. Run the Forecasting Pipeline

```bash
python src/forecast_next.py
```

The pipeline produces:

* Current load
* Next-hour forecast
* Peak probability
* Peak warning
* Peak threshold
* AI-generated energy insight

### 8. Run Workload Simulation

```bash
python src/workload_shift.py
```

---

## 🔐 Environment Variables

Create a local `.env` file:

```env
GROQ_API_KEY=your_api_key_here
```

Make sure `.env` is included in `.gitignore:

```gitignore
.env
venv/
__pycache__/
*.pyc
```

---

## 📌 Example Output

### Next-Hour Energy Forecast

```text
Current time: 2014-12-31 23:00:01
Forecast time: 2015-01-01 00:00:01

Current load: 11.0 kW
Predicted load: 11.32 kW

Peak probability: 0.04%
Peak threshold: 72.0 kW
Peak warning: NO
```

### AI Energy Assistant

```text
Expected demand:
The load is projected to rise slightly from 11.00 kW
to 11.32 kW in the next hour.

Peak-demand assessment:
The upcoming hour is not a peak-demand period.

Recommendation:
No immediate peak-management action is indicated.
```

---

## 🎯 What This Project Demonstrates

This project combines several practical ML and GenAI concepts:

* Time-series feature engineering
* Lag-based forecasting
* Gradient-boosting regression
* Imbalanced classification
* Peak-demand detection
* What-if simulation
* Model evaluation against a baseline
* ML model persistence with Joblib
* REST/API-based LLM integration
* Prompt grounding
* GenAI-assisted interpretation of ML outputs

---

## ⚠️ Limitations

* The current implementation uses **one representative electricity time series** for the forecasting model.
* Weather conditions are not included.
* Electricity pricing is not included.
* Holidays and special events are not explicitly modeled.
* Workload shifting is a simulated scenario rather than real-world control.
* The GenAI assistant only receives the numerical forecasting and simulation results supplied by the pipeline.
* The system is intended for **experimentation and demonstration**, not operational electricity-grid control.

---

## 📚 Dataset Source

**Electricity Load Diagrams 2011–2014 — UCI Machine Learning Repository**

Dataset DOI:

```text
10.24432/C58C86
```

The hourly aggregated data used in this project is derived from the original electricity load dataset.

---

## 👩‍💻 Author

**Gayathri Gunturi**

Built as an end-to-end **Machine Learning + Generative AI** project for electricity demand forecasting and energy analytics.
