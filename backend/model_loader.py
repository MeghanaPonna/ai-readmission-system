"""
model_loader.py
---------------
Loads the trained ML model and preprocessing artifacts (encoders,
imputer, scaler, feature names) from the ml/artifacts/ directory.
"""

import os
import joblib
from typing import Any

# Resolve path relative to this file's location
_BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
_ARTIFACTS_DIR = os.path.join(_BASE_DIR, "..", "ml", "artifacts")


def load_all_artifacts() -> dict[str, Any]:
    """
    Load and return a dict containing:
      - model         : trained sklearn/XGBoost classifier
      - encoders      : dict of LabelEncoders keyed by column name
      - imputer       : fitted SimpleImputer
      - scaler        : fitted StandardScaler
      - feature_names : ordered list of feature column names
    """
    def _load(filename: str) -> Any:
        path = os.path.join(_ARTIFACTS_DIR, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Artifact not found: {path}\n"
                "Run  python ml/train_model.py  first."
            )
        return joblib.load(path)

    return {
        "model":         _load("best_model.joblib"),
        "encoders":      _load("encoders.joblib"),
        "imputer":       _load("imputer.joblib"),
        "scaler":        _load("scaler.joblib"),
        "feature_names": _load("feature_names.joblib"),
    }
