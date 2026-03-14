# AI Patient Readmission Prediction System

An **end-to-end Machine Learning + Full Stack application** that predicts whether a hospital patient will be **readmitted within 30 days after discharge** using clinical data.

The system analyzes patient health records and generates a **risk score and recommendation** to help healthcare providers identify **high-risk patients early**.

---

## Live Demo

Frontend:
[https://ai-readmission-system.vercel.app/](https://ai-readmission-system.vercel.app/)

API Docs:
[https://ai-readmission-system-1.onrender.com/docs](https://ai-readmission-system-1.onrender.com/docs)

---

# Key Features

* Predicts **30-day hospital readmission risk**
* Machine learning model trained on **100,000+ patient records**
* **XGBoost** model for prediction
* **FastAPI backend** for real-time predictions
* **React dashboard** for patient data input
* Risk classification: **Low / Medium / High**

---

# Tech Stack

### Machine Learning

* Python
* Pandas
* Scikit-learn
* XGBoost

### Backend

* FastAPI
* Uvicorn
* Joblib

### Frontend

* React
* JavaScript
* CSS

---

# Dataset

The model is trained on the **Diabetes 130-US Hospitals Dataset**.

* **101,766 patient encounters**
* **47 clinical features**
* Target variable: **30-day readmission**

Dataset Source:
[https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008](https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008)

---

# System Architecture

```
React Frontend
      ↓
FastAPI Backend
      ↓
Preprocessing Pipeline
      ↓
XGBoost Model
      ↓
Readmission Risk Prediction
```

---

# Project Structure

```
ai-readmission-system
│
├── ml/          # Machine learning pipeline
├── backend/     # FastAPI backend
├── frontend/    # React application
└── README.md
```

---

# Running the Project

### 1. Train the Model

```
cd ml
python train_model.py
```

---

### 2. Start Backend API

```
cd backend
uvicorn main:app --reload --port 8000
```

API documentation:

```
http://localhost:8000/docs
```

---

### 3. Run Frontend

```
cd frontend
npm install
npm start
```

Open in browser:

```
http://localhost:3000
```

---

# Model Performance

**Best Model:** XGBoost

* Accuracy: **~82%**
* ROC-AUC: **~0.68**

Important predictive features include:

* Previous inpatient visits
* Emergency visits
* Length of hospital stay
* Number of medications
* Diagnosis type

---

# Disclaimer

This project is intended for **educational and research purposes only** and should **not replace professional medical judgment**.
