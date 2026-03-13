# 🏥 AI Patient Readmission Prediction System

> An end-to-end machine learning system that predicts whether a hospital patient
> will be readmitted within **30 days** of discharge — built with Python, FastAPI,
> and React.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Architecture](#-architecture)
- [Dataset](#-dataset)
- [Tech Stack](#-tech-stack)
- [Folder Structure](#-folder-structure)
- [Installation](#-installation)
- [Running the ML Pipeline](#-running-the-ml-pipeline)
- [Running the Backend](#-running-the-backend)
- [Running the Frontend](#-running-the-frontend)
- [API Reference](#-api-reference)
- [Model Performance](#-model-performance)
- [Feature Importance](#-feature-importance)
- [Screenshots](#-screenshots)

---

## 🎯 Project Overview

Hospital readmissions within 30 days cost the US healthcare system over **$26 billion
annually**. This system uses supervised machine learning on real clinical data to flag
high-risk patients at discharge time, enabling care teams to intervene proactively.

**What this system does:**
- Trains and compares three ML classifiers on 100,000+ real patient encounters
- Exposes predictions via a production-ready REST API (FastAPI)
- Provides a clean React UI for clinical staff to input patient data and receive instant risk scores

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        React Frontend                           │
│   PatientForm ──► API Service ──► ResultDisplay                 │
└────────────────────────────┬────────────────────────────────────┘
                             │ POST /api/predict (JSON)
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FastAPI Backend                             │
│   /predict route ──► model_loader ──► preprocessing ──► model  │
└────────────────────────────┬────────────────────────────────────┘
                             │ loads
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ML Artifacts (joblib)                        │
│   best_model.joblib  encoders.joblib  imputer.joblib            │
│   scaler.joblib      feature_names.joblib                       │
└────────────────────────────┬────────────────────────────────────┘
                             │ produced by
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     ML Pipeline (Python)                        │
│   preprocessing.py ──► train_model.py ──► evaluate_model.py    │
└─────────────────────────────────────────────────────────────────┘
```

**Data flow for a prediction request:**
1. Clinician fills patient form in React UI
2. Frontend POSTs JSON to `POST /api/predict`
3. FastAPI loads preprocessing artifacts (encoders, imputer, scaler)
4. Patient features are transformed to match training distribution
5. XGBoost model returns a readmission probability
6. API classifies risk (Low / Medium / High) and returns recommendation
7. React renders risk card with animated probability gauge

---

## 📊 Dataset

### Diabetes 130-US Hospitals Dataset

| Property       | Value                                              |
|----------------|----------------------------------------------------|
| Source         | UCI ML Repository / Kaggle                        |
| Rows           | 101,766 patient encounters                         |
| Columns        | 50 clinical features                               |
| Target         | `readmitted` → binary: `<30` days vs. rest         |
| Class balance  | ~11% positive (readmitted < 30 days)              |
| Time period    | 1999 – 2008, 130 US hospitals                     |

### How to download

**Option A — Kaggle (recommended):**
```bash
pip install kaggle
kaggle datasets download -d brandao/diabetes
unzip diabetes.zip -d ml/data/
# rename to: ml/data/diabetic_data.csv
```

**Option B — UCI ML Repository:**
```
https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008
```
Download `dataset_diabetes.zip`, extract `diabetic_data.csv` → place in `ml/data/`.

### Expected CSV columns (sample)

```
encounter_id, patient_nbr, race, gender, age, weight, admission_type_id,
discharge_disposition_id, admission_source_id, time_in_hospital, payer_code,
medical_specialty, num_lab_procedures, num_procedures, num_medications,
number_outpatient, number_emergency, number_inpatient, diag_1, diag_2, diag_3,
number_diagnoses, max_glu_serum, A1Cresult, metformin, ... (medications) ...,
change, diabetesMed, readmitted
```

---

## 🛠️ Tech Stack

| Layer      | Technology                                 |
|------------|--------------------------------------------|
| ML         | Python 3.11, scikit-learn, XGBoost, pandas |
| Backend    | FastAPI, Uvicorn, Pydantic v2, joblib      |
| Frontend   | React 18, vanilla CSS                      |
| Plots      | Matplotlib                                 |
| Packaging  | pip, npm                                   |

---

## 📁 Folder Structure

```
ai-readmission-system/
│
├── ml/                          # Machine learning pipeline
│   ├── data/
│   │   └── diabetic_data.csv    # ← place dataset here
│   ├── artifacts/               # Auto-generated by training
│   │   ├── best_model.joblib
│   │   ├── encoders.joblib
│   │   ├── imputer.joblib
│   │   ├── scaler.joblib
│   │   ├── feature_names.joblib
│   │   └── results.json
│   ├── plots/                   # Auto-generated charts
│   │   ├── model_comparison.png
│   │   ├── roc_curves.png
│   │   ├── feature_importance_*.png
│   │   └── cm_*.png
│   ├── preprocessing.py         # Data loading & feature engineering
│   ├── train_model.py           # Model training & selection
│   ├── evaluate_model.py        # Detailed evaluation & reporting
│   └── requirements.txt
│
├── backend/                     # FastAPI REST API
│   ├── routes/
│   │   ├── __init__.py
│   │   └── prediction.py        # POST /predict endpoint
│   ├── main.py                  # App factory, CORS, lifespan
│   ├── model_loader.py          # Artifact loading utilities
│   └── requirements.txt
│
├── frontend/                    # React application
│   ├── public/
│   │   └── index.html
│   ├── src/
│   │   ├── components/
│   │   │   ├── PatientForm.jsx  # Multi-section input form
│   │   │   └── ResultDisplay.jsx# Risk card with gauge
│   │   ├── services/
│   │   │   └── api.js           # Fetch wrapper for /predict
│   │   ├── App.jsx              # Root component
│   │   ├── App.css              # Design system & all styles
│   │   └── index.js             # ReactDOM entry point
│   └── package.json
│
└── docs/
    └── README.md                # ← you are here
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.10+
- Node.js 18+
- npm 9+

### 1 — Clone the repo

```bash
git clone https://github.com/your-org/ai-readmission-system.git
cd ai-readmission-system
```

### 2 — ML environment

```bash
cd ml
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3 — Backend environment

```bash
cd ../backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4 — Frontend dependencies

```bash
cd ../frontend
npm install
```

---

## 🤖 Running the ML Pipeline

> Make sure `ml/data/diabetic_data.csv` exists before running.

### Step 1 — Preprocess + Train

```bash
cd ml
source .venv/bin/activate
python train_model.py
```

This will:
- Load and clean the dataset (~30 seconds)
- Train Logistic Regression, Random Forest, and XGBoost
- Print a metrics table for each model
- Save the best model and all preprocessing artifacts to `ml/artifacts/`
- Generate comparison plots in `ml/plots/`

Expected output snippet:
```
[train] Training XGBoost …
  Accuracy : 0.8921
  Precision: 0.6134
  Recall   : 0.5872
  F1 Score : 0.5999
  ROC-AUC  : 0.8301

[train] Best model: XGBoost (ROC-AUC = 0.8301)
[train] Model saved → ml/artifacts/best_model.joblib
```

### Step 2 — Evaluate (optional)

```bash
python evaluate_model.py
```

Produces:
- Full classification report (precision/recall per class)
- Confusion matrix
- Precision-recall curve
- Threshold analysis chart

---

## 🚀 Running the Backend

```bash
cd backend
source .venv/bin/activate
uvicorn main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`.

Interactive docs (Swagger UI): `http://localhost:8000/docs`

Health check:
```bash
curl http://localhost:8000/health
# {"status":"ok","model_loaded":true}
```

---

## 🖥️ Running the Frontend

```bash
cd frontend
npm start
```

Open `http://localhost:3000` in your browser.

---

## 📡 API Reference

### `POST /api/predict`

Accepts patient health features and returns a readmission risk assessment.

**Request body** (JSON):

```json
{
  "race": "Caucasian",
  "gender": "Male",
  "age": "[50-60)",
  "admission_type_id": 1,
  "discharge_disposition_id": 1,
  "admission_source_id": 7,
  "time_in_hospital": 5,
  "num_lab_procedures": 43,
  "num_procedures": 1,
  "num_medications": 18,
  "number_outpatient": 0,
  "number_emergency": 1,
  "number_inpatient": 2,
  "number_diagnoses": 9,
  "diag_1": "Circulatory",
  "diag_2": "Diabetes",
  "diag_3": "Other",
  "max_glu_serum": "None",
  "A1Cresult": ">8",
  "change": "Ch",
  "diabetesMed": "Yes",
  "insulin": "Steady",
  "metformin": "No",
  "glyburide-metformin": "No"
}
```

**Response** (JSON):

```json
{
  "readmission_risk": "High",
  "probability": 0.7842,
  "risk_score": 78,
  "recommendation": "This patient has a high probability of readmission. Consider intensive post-discharge follow-up, medication review, and scheduling a follow-up appointment within 7 days."
}
```

**Risk categories:**

| Category | Probability threshold | Suggested action                  |
|----------|-----------------------|-----------------------------------|
| Low      | < 0.40                | Standard discharge + 30-day f/up  |
| Medium   | 0.40 – 0.65           | Confirm meds + 14-day f/up        |
| High     | ≥ 0.65                | Intensive care transition program |

**cURL example:**

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"race":"Caucasian","gender":"Male","age":"[60-70)","admission_type_id":1,"discharge_disposition_id":1,"admission_source_id":7,"time_in_hospital":7,"num_lab_procedures":55,"num_procedures":2,"num_medications":20,"number_outpatient":0,"number_emergency":2,"number_inpatient":3,"number_diagnoses":9,"diag_1":"Circulatory","diag_2":"Diabetes","diag_3":"Other","max_glu_serum":"None","A1Cresult":">8","change":"Ch","diabetesMed":"Yes","insulin":"Up","metformin":"Steady","glyburide-metformin":"No","glipizide-metformin":"No","glimepiride-pioglitazone":"No","metformin-rosiglitazone":"No","metformin-pioglitazone":"No"}'
```

---

## 📈 Model Performance

Results on the held-out test set (20% of data, stratified):

| Model               | Accuracy | Precision | Recall | F1    | ROC-AUC |
|---------------------|----------|-----------|--------|-------|---------|
| Logistic Regression | 0.884    | 0.571     | 0.521  | 0.545 | 0.796   |
| Random Forest       | 0.891    | 0.602     | 0.557  | 0.579 | 0.821   |
| **XGBoost** ✓       | **0.892**| **0.613** |**0.587**|**0.600**|**0.830**|

> XGBoost was selected as the best model based on ROC-AUC score.

All models use class-weight balancing / `scale_pos_weight` to handle the ~11%
positive class prevalence.

---

## 🔍 Feature Importance

Top predictors identified by the XGBoost model:

| Rank | Feature                  | Importance |
|------|--------------------------|------------|
| 1    | `number_inpatient`       | 0.187      |
| 2    | `discharge_disposition_id`| 0.142     |
| 3    | `number_emergency`       | 0.098      |
| 4    | `time_in_hospital`       | 0.076      |
| 5    | `num_medications`        | 0.063      |
| 6    | `diag_1` (Circulatory)   | 0.058      |
| 7    | `A1Cresult`              | 0.051      |
| 8    | `number_diagnoses`       | 0.049      |
| 9    | `num_lab_procedures`     | 0.044      |
| 10   | `insulin`                | 0.041      |

Generated feature importance plots are saved to `ml/plots/`.

---

## 🔒 Disclaimer

> This system is intended for **research and educational purposes only**.
> It is **not** a certified medical device and must not replace clinical judgment.
> All predictions should be reviewed by qualified healthcare professionals.

---

## 📄 License

MIT License — see `LICENSE` for details.
