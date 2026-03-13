"""
routes/prediction.py
--------------------
POST /predict  — accepts patient data, returns readmission risk.
"""

from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np

router = APIRouter()


# ── Input schema ───────────────────────────────────────────────────────────────
class PatientData(BaseModel):
    """
    Patient health features expected by the model.
    All categorical fields accept string values that match the
    original dataset; unknown values are handled gracefully.
    """
    # Demographics
    race: str                  = Field("Caucasian", example="Caucasian")
    gender: str                = Field("Male",      example="Male")
    age: str                   = Field("[50-60)",   example="[50-60)")

    # Admission info
    admission_type_id: int     = Field(1,  ge=1, le=8)
    discharge_disposition_id: int = Field(1, ge=1, le=30)
    admission_source_id: int   = Field(7,  ge=1, le=26)

    # Hospital stay metrics
    time_in_hospital: int      = Field(3,   ge=1,  le=14)
    num_lab_procedures: int    = Field(43,  ge=0,  le=132)
    num_procedures: int        = Field(1,   ge=0,  le=6)
    num_medications: int       = Field(15,  ge=0,  le=81)
    number_outpatient: int     = Field(0,   ge=0)
    number_emergency: int      = Field(0,   ge=0)
    number_inpatient: int      = Field(0,   ge=0)
    number_diagnoses: int      = Field(9,   ge=1,  le=16)

    # Diagnosis categories (already simplified; raw ICD-9 also accepted)
    diag_1: str                = Field("Circulatory", example="Circulatory")
    diag_2: str                = Field("Circulatory", example="Diabetes")
    diag_3: str                = Field("Other",       example="Other")

    # Lab results
    max_glu_serum: str         = Field("None", example="None")
    A1Cresult: str             = Field("None", example="None")

    # Medications
    metformin: str             = Field("No",  example="No")
    repaglinide: str           = Field("No",  example="No")
    nateglinide: str           = Field("No",  example="No")
    chlorpropamide: str        = Field("No",  example="No")
    glimepiride: str           = Field("No",  example="No")
    acetohexamide: str         = Field("No",  example="No")
    glipizide: str             = Field("No",  example="No")
    glyburide: str             = Field("No",  example="No")
    tolbutamide: str           = Field("No",  example="No")
    pioglitazone: str          = Field("No",  example="No")
    rosiglitazone: str         = Field("No",  example="No")
    acarbose: str              = Field("No",  example="No")
    miglitol: str              = Field("No",  example="No")
    troglitazone: str          = Field("No",  example="No")
    tolazamide: str            = Field("No",  example="No")
    insulin: str               = Field("No",  example="No")
    change: str                = Field("No",  example="No")
    diabetesMed: str           = Field("Yes", example="Yes")

    # Combined meds (often "No" / "Steady" / "Up" / "Down")
    glyburide_metformin: str          = Field("No", alias="glyburide-metformin")
    glipizide_metformin: str          = Field("No", alias="glipizide-metformin")
    glimepiride_pioglitazone: str     = Field("No", alias="glimepiride-pioglitazone")
    metformin_rosiglitazone: str      = Field("No", alias="metformin-rosiglitazone")
    metformin_pioglitazone: str       = Field("No", alias="metformin-pioglitazone")

    class Config:
        populate_by_name = True


# ── Output schema ──────────────────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    readmission_risk: str
    probability: float
    risk_score: int          # 0-100 integer for UI progress bars
    recommendation: str


# ── Risk thresholds ────────────────────────────────────────────────────────────
def _classify_risk(prob: float) -> tuple[str, str]:
    if prob >= 0.65:
        return "High", (
            "This patient has a high probability of readmission. "
            "Consider intensive post-discharge follow-up, medication review, "
            "and scheduling a follow-up appointment within 7 days."
        )
    elif prob >= 0.40:
        return "Medium", (
            "Moderate readmission risk detected. "
            "Ensure clear discharge instructions, confirm medication adherence "
            "support, and schedule a follow-up within 14 days."
        )
    else:
        return "Low", (
            "Low readmission risk. "
            "Standard discharge protocol and a 30-day follow-up appointment "
            "are recommended."
        )


# ── Preprocessing helper (mirrors ml/preprocessing.py logic) ───────────────────
_NUMERIC_COLS = [
    "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses"
]

# Map Pydantic field names back to original column names with hyphens
_ALIAS_MAP = {
    "glyburide_metformin":      "glyburide-metformin",
    "glipizide_metformin":      "glipizide-metformin",
    "glimepiride_pioglitazone": "glimepiride-pioglitazone",
    "metformin_rosiglitazone":  "metformin-rosiglitazone",
    "metformin_pioglitazone":   "metformin-pioglitazone",
}


def _build_dataframe(patient: PatientData) -> pd.DataFrame:
    """Convert Pydantic model → one-row DataFrame with original column names."""
    raw = patient.dict(by_alias=False)
    # Restore hyphenated column names
    renamed = {}
    for k, v in raw.items():
        renamed[_ALIAS_MAP.get(k, k)] = v
    return pd.DataFrame([renamed])


def _preprocess_input(df: pd.DataFrame, artifacts: dict) -> np.ndarray:
    """Apply saved encoders, imputer, and scaler to a single-row DataFrame."""
    encoders      = artifacts["encoders"]
    imputer       = artifacts["imputer"]
    scaler        = artifacts["scaler"]
    feature_names = artifacts["feature_names"]

    # Encode categorical columns
    for col, le in encoders.items():
        if col in df.columns:
            val = str(df[col].iloc[0])
            if val not in set(le.classes_):
                val = le.classes_[0]  # fallback to first known class
            df[col] = le.transform([val])

    # Impute + scale numeric columns
    num_cols = [c for c in _NUMERIC_COLS if c in df.columns]
    df[num_cols] = imputer.transform(df[num_cols])
    df[num_cols] = scaler.transform(df[num_cols])

    # Reorder columns to match training order
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0          # fill any missing feature with 0
    return df[feature_names].values


# ── Route ──────────────────────────────────────────────────────────────────────
@router.post("/predict", response_model=PredictionResponse)
async def predict(patient: PatientData, request: Request):
    """
    Accept patient health features and return:
    - readmission_risk: "Low" | "Medium" | "High"
    - probability: float between 0 and 1
    - risk_score: integer 0-100
    - recommendation: clinical guidance string
    """
    try:
        artifacts = request.app.state.artifacts
        df        = _build_dataframe(patient)
        X         = _preprocess_input(df, artifacts)

        model = artifacts["model"]
        prob  = float(model.predict_proba(X)[0, 1])

        risk, recommendation = _classify_risk(prob)
        return PredictionResponse(
            readmission_risk=risk,
            probability=round(prob, 4),
            risk_score=int(prob * 100),
            recommendation=recommendation,
        )

    except Exception as exc:
        raise HTTPException(status_code=500,
                            detail=f"Prediction failed: {str(exc)}")
