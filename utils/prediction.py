"""Prediction functions for ClaimIQ."""
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


def predict_glm(
    model: Any,
    driv_age: float,
    veh_age: float,
    bonus_malus: float,
    density: float,
    exposure: float,
) -> Tuple[Optional[float], Optional[str]]:
    """
    Generate a prediction from a statsmodels GLM result.

    The GLM was trained with log(Exposure) as an offset term.
    The prediction represents expected claim count for the given exposure period.
    """
    try:
        data = pd.DataFrame({
            "BonusMalus": [float(bonus_malus)],
            "DrivAge":    [float(driv_age)],
            "VehAge":     [float(veh_age)],
            "Density":    [float(density)],
            "Exposure":   [float(exposure)],
        })
        offset = np.log(data["Exposure"])
        pred = model.predict(data, offset=offset)
        value = max(float(pred.iloc[0]), 0.0)
        return value, None
    except Exception as exc:
        return None, f"GLM prediction failed: {exc}"


def predict_sklearn(
    model: Any,
    driv_age: float,
    veh_age: float,
    bonus_malus: float,
    density: float,
    exposure: float,
    feature_order: list,
) -> Tuple[Optional[float], Optional[str]]:
    """
    Generate a prediction from a scikit-learn or XGBoost model.

    Features must be passed in the exact order used during training.
    """
    try:
        data = pd.DataFrame({
            "Exposure":   [float(exposure)],
            "BonusMalus": [float(bonus_malus)],
            "DrivAge":    [float(driv_age)],
            "VehAge":     [float(veh_age)],
            "Density":    [float(density)],
        })
        X = data[feature_order]
        pred = model.predict(X)
        value = max(float(pred[0]), 0.0)
        return value, None
    except Exception as exc:
        return None, f"Model prediction failed: {exc}"


def compute_annualized_frequency(predicted_claims: float, exposure: float) -> float:
    """Convert predicted claim count to annualized frequency."""
    if exposure <= 0:
        return 0.0
    return predicted_claims / exposure


def classify_risk(annualized_frequency: float) -> Dict[str, str]:
    """
    Classify annualized claim frequency into an illustrative risk tier.

    Thresholds are based on the empirical distribution of the French MTPL dataset.
    These are illustrative and not official underwriting classifications.
    """
    LOW_THRESHOLD      = 0.04
    MODERATE_THRESHOLD = 0.08

    if annualized_frequency < LOW_THRESHOLD:
        return {"level": "Low", "color": "green", "description": "Below 4th percentile threshold"}
    elif annualized_frequency < MODERATE_THRESHOLD:
        return {"level": "Moderate", "color": "orange", "description": "Between low and high thresholds"}
    else:
        return {"level": "High", "color": "red", "description": "Above 8th percentile threshold"}


def compute_pure_premium(predicted_claims: float, avg_claim_cost: float) -> float:
    """Compute illustrative pure premium for the selected exposure period."""
    return predicted_claims * avg_claim_cost


def compute_loaded_premium(
    pure_premium: float,
    expense_ratio: float,
    margin: float,
) -> Optional[float]:
    """
    Compute illustrative loaded (gross) premium.

    Formula: Pure Premium / (1 - expense_ratio - margin)
    """
    denominator = 1.0 - expense_ratio - margin
    if denominator <= 0:
        return None
    return pure_premium / denominator


def scenario_predictions(
    model: Any,
    model_type: str,
    vary_var: str,
    vary_values: np.ndarray,
    fixed: Dict[str, float],
    feature_order: list,
) -> np.ndarray:
    """
    Generate predictions across a range of values for one variable.

    Parameters
    ----------
    model      : fitted model object
    model_type : 'glm' or 'sklearn'
    vary_var   : variable name being varied
    vary_values: array of values to iterate over
    fixed      : dict of fixed values for all other variables
    feature_order : feature column order for sklearn models
    """
    results = []
    for v in vary_values:
        params = {**fixed, vary_var: v}
        driv_age    = params.get("DrivAge",    35.0)
        veh_age     = params.get("VehAge",     5.0)
        bonus_malus = params.get("BonusMalus", 70.0)
        density     = params.get("Density",    1500.0)
        exposure    = params.get("Exposure",   1.0)

        if model_type == "glm":
            pred, err = predict_glm(model, driv_age, veh_age, bonus_malus, density, exposure)
        else:
            pred, err = predict_sklearn(model, driv_age, veh_age, bonus_malus, density, exposure, feature_order)

        if pred is not None:
            results.append(compute_annualized_frequency(pred, exposure))
        else:
            results.append(np.nan)

    return np.array(results)
