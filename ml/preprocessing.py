"""
preprocessing.py
----------------
Data loading and preprocessing pipeline for the Patient Readmission
Prediction System using the Diabetes 130-US Hospitals dataset.

Dataset source:
  UCI ML Repository: https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008
  Kaggle mirror:     https://www.kaggle.com/datasets/brandao/diabetes
  File to download:  diabetic_data.csv  (≈ 101,766 rows × 50 columns)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import os

# ── Constants ──────────────────────────────────────────────────────────────────
DATA_PATH      = os.path.join(os.path.dirname(__file__), "data", "diabetic_data.csv")
ARTIFACTS_DIR  = os.path.join(os.path.dirname(__file__), "artifacts")
RANDOM_STATE   = 42

# Columns that encode "no info" as '?' in the raw CSV
MISSING_MARKER = "?"

# High-cardinality / administrative columns to drop
DROP_COLS = [
    "encounter_id", "patient_nbr", "examide", "citoglipton",
    "weight",           # >96 % missing
    "payer_code",       # not clinically relevant
    "medical_specialty" # >50 % missing
]

# Medication columns (binary change indicators we keep as-is after encoding)
MED_COLS = [
    "metformin","repaglinide","nateglinide","chlorpropamide","glimepiride",
    "acetohexamide","glipizide","glyburide","tolbutamide","pioglitazone",
    "rosiglitazone","acarbose","miglitol","troglitazone","tolazamide",
    "insulin","glyburide-metformin","glipizide-metformin",
    "glimepiride-pioglitazone","metformin-rosiglitazone","metformin-pioglitazone"
]

CATEGORICAL_COLS = [
    "race", "gender", "age",
    "admission_type_id", "discharge_disposition_id", "admission_source_id",
    "diag_1", "diag_2", "diag_3",
    "max_glu_serum", "A1Cresult",
    "change", "diabetesMed"
] + MED_COLS

NUMERIC_COLS = [
    "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses"
]

TARGET_COL = "readmitted"


# ── Helper functions ───────────────────────────────────────────────────────────

def load_raw_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load raw CSV and replace '?' sentinel with NaN."""
    print(f"[preprocessing] Loading data from: {path}")
    df = pd.read_csv(path, na_values=MISSING_MARKER)
    print(f"[preprocessing] Raw shape: {df.shape}")
    return df


def engineer_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert multi-class readmission label to binary:
      1  → patient readmitted within 30 days  (positive class)
      0  → not readmitted or readmitted >30 days
    """
    df = df.copy()
    df[TARGET_COL] = (df[TARGET_COL] == "<30").astype(int)
    print(f"[preprocessing] Target distribution:\n{df[TARGET_COL].value_counts()}")
    return df


def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove administrative and high-missing-rate columns."""
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    print(f"[preprocessing] Dropped {len(cols_to_drop)} columns → shape: {df.shape}")
    return df


def simplify_diagnosis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map ICD-9 diagnosis codes to high-level disease categories
    to reduce cardinality of diag_1/2/3.
    """
    def icd9_to_category(code):
        if pd.isna(code):
            return "Unknown"
        code = str(code)
        if code.startswith("V") or code.startswith("E"):
            return "External"
        try:
            num = float(code)
        except ValueError:
            return "Other"
        if 390 <= num <= 459 or num == 785:
            return "Circulatory"
        elif 460 <= num <= 519 or num == 786:
            return "Respiratory"
        elif 520 <= num <= 579 or num == 787:
            return "Digestive"
        elif 250 <= num <= 250.99:
            return "Diabetes"
        elif 800 <= num <= 999:
            return "Injury"
        elif 710 <= num <= 739:
            return "Musculoskeletal"
        elif 580 <= num <= 629 or num == 788:
            return "Genitourinary"
        elif 140 <= num <= 239:
            return "Neoplasms"
        else:
            return "Other"

    for col in ["diag_1", "diag_2", "diag_3"]:
        if col in df.columns:
            df[col] = df[col].apply(icd9_to_category)
    return df


def encode_categoricals(df: pd.DataFrame,
                         encoders: dict = None,
                         fit: bool = True) -> tuple:
    """
    Label-encode all categorical columns.
    Returns (transformed_df, encoder_dict).
    """
    if encoders is None:
        encoders = {}

    df = df.copy()
    cat_cols = [c for c in CATEGORICAL_COLS if c in df.columns]

    # Fill remaining NaN in categorical cols with 'Unknown'
    df[cat_cols] = df[cat_cols].fillna("Unknown").astype(str)

    for col in cat_cols:
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
        else:
            le = encoders[col]
            # Handle unseen labels gracefully
            known = set(le.classes_)
            df[col] = df[col].apply(lambda x: x if x in known else le.classes_[0])
            df[col] = le.transform(df[col])

    return df, encoders


def impute_and_scale(df: pd.DataFrame,
                     imputer: SimpleImputer = None,
                     scaler: StandardScaler = None,
                     fit: bool = True) -> tuple:
    """
    Impute missing numeric values with median then standard-scale.
    Returns (transformed_df, imputer, scaler).
    """
    df = df.copy()
    num_cols = [c for c in NUMERIC_COLS if c in df.columns]

    if fit:
        imputer = SimpleImputer(strategy="median")
        scaler  = StandardScaler()
        df[num_cols] = imputer.fit_transform(df[num_cols])
        df[num_cols] = scaler.fit_transform(df[num_cols])
    else:
        df[num_cols] = imputer.transform(df[num_cols])
        df[num_cols] = scaler.transform(df[num_cols])

    return df, imputer, scaler


def preprocess(path: str = DATA_PATH,
               test_size: float = 0.2,
               save_artifacts: bool = True):
    """
    Full preprocessing pipeline.

    Returns
    -------
    X_train, X_test, y_train, y_test, feature_names
    """
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # 1. Load
    df = load_raw_data(path)

    # 2. Binary target
    df = engineer_target(df)

    # 3. Drop irrelevant columns
    df = drop_columns(df)

    # 4. Simplify ICD-9 codes
    df = simplify_diagnosis(df)

    # 5. Separate features and target
    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])

    # 6. Train/test split (stratified to preserve class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )

    # 7. Encode categoricals (fit on train only)
    X_train, encoders = encode_categoricals(X_train, fit=True)
    X_test,  _        = encode_categoricals(X_test,  encoders=encoders, fit=False)

    # 8. Impute + scale numerics (fit on train only)
    X_train, imputer, scaler = impute_and_scale(X_train, fit=True)
    X_test,  _,       _      = impute_and_scale(X_test, imputer=imputer,
                                                 scaler=scaler, fit=False)

    feature_names = list(X_train.columns)
    print(f"[preprocessing] Final feature count: {len(feature_names)}")
    print(f"[preprocessing] Train size: {X_train.shape}, Test size: {X_test.shape}")

    # 9. Persist artifacts
    if save_artifacts:
        joblib.dump(encoders, os.path.join(ARTIFACTS_DIR, "encoders.joblib"))
        joblib.dump(imputer,  os.path.join(ARTIFACTS_DIR, "imputer.joblib"))
        joblib.dump(scaler,   os.path.join(ARTIFACTS_DIR, "scaler.joblib"))
        joblib.dump(feature_names,
                    os.path.join(ARTIFACTS_DIR, "feature_names.joblib"))
        print(f"[preprocessing] Artifacts saved to {ARTIFACTS_DIR}/")

    return X_train, X_test, y_train, y_test, feature_names


# ── Standalone entry point ─────────────────────────────────────────────────────
if __name__ == "__main__":
    preprocess()
