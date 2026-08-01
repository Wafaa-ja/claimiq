"""Input validation functions for ClaimIQ."""
from typing import Tuple, List


TRAINING_RANGES = {
    "DrivAge":    (18, 100),
    "VehAge":     (0, 100),
    "BonusMalus": (50, 230),
    "Density":    (1, 27000),
    "Exposure":   (0.01, 2.00),
}

BOUNDARY_MARGIN = 0.05  # warn if within 5% of edge


def validate_inputs(
    driv_age: int,
    veh_age: int,
    bonus_malus: int,
    density: int,
    exposure: float,
) -> Tuple[List[str], List[str]]:
    """
    Validate user inputs against training data ranges.

    Returns
    -------
    errors : list of str
        Hard validation failures that block prediction.
    warnings : list of str
        Soft warnings shown to the user but not blocking.
    """
    errors: List[str] = []
    warnings: List[str] = []

    inputs = {
        "DrivAge": driv_age,
        "VehAge": veh_age,
        "BonusMalus": bonus_malus,
        "Density": density,
        "Exposure": exposure,
    }

    for name, value in inputs.items():
        lo, hi = TRAINING_RANGES[name]
        if value < lo or value > hi:
            errors.append(
                f"{name} value of {value} is outside the supported range "
                f"({lo}–{hi}). Predictions outside this range may be unreliable."
            )
        else:
            span = hi - lo
            if span > 0:
                pct_from_lo = (value - lo) / span
                pct_from_hi = (hi - value) / span
                if pct_from_lo < BOUNDARY_MARGIN or pct_from_hi < BOUNDARY_MARGIN:
                    warnings.append(
                        f"{name} = {value} is near the boundary of the training data. "
                        "Predictions may be less reliable at this value."
                    )

    if exposure <= 0:
        errors.append("Exposure must be greater than zero.")

    return errors, warnings


def validate_premium_inputs(
    avg_claim_cost: float,
    expense_ratio: float,
    margin: float,
) -> List[str]:
    """Validate pure premium calculator inputs."""
    errors: List[str] = []

    if avg_claim_cost < 0:
        errors.append("Average claim cost must be non-negative.")

    if not (0 <= expense_ratio < 1):
        errors.append("Expense ratio must be between 0% and 99%.")

    if not (0 <= margin < 1):
        errors.append("Risk/profit margin must be between 0% and 99%.")

    if expense_ratio + margin >= 1:
        errors.append(
            "The sum of expense ratio and margin must be less than 100%. "
            "Otherwise the denominator becomes zero or negative."
        )

    return errors
