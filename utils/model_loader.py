"""Model loading utilities for ClaimIQ."""
import json
import os
from typing import Any, Dict, Optional, Tuple

import joblib
import statsmodels.api as sm


MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def load_glm_model(filename: str) -> Tuple[Optional[Any], Optional[str]]:
    """
    Load a statsmodels GLM result from disk.

    Returns (model, error_message). If loading fails, model is None.
    """
    path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(path):
        return None, f"Model file not found: {path}"
    try:
        model = sm.load(path)
        return model, None
    except Exception as exc:
        return None, f"Failed to load {filename}: {exc}"


def load_sklearn_model(filename: str) -> Tuple[Optional[Any], Optional[str]]:
    """
    Load a scikit-learn or XGBoost model from disk using joblib.

    Returns (model, error_message). If loading fails, model is None.
    """
    path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(path):
        return None, f"Model file not found: {path}"
    try:
        model = joblib.load(path)
        return model, None
    except Exception as exc:
        return None, f"Failed to load {filename}: {exc}"


def load_feature_metadata() -> Dict:
    """Load feature metadata JSON from the models directory."""
    path = os.path.join(MODELS_DIR, "feature_metadata.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)
