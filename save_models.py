"""
save_models.py
==============
Trains and saves all four ClaimIQ models from the French MTPL dataset.

Usage
-----
    python save_models.py --data path/to/freMTPL2freq.csv

The script will:
1. Load and validate the dataset.
2. Apply the same train/test split (random_state=42, test_size=0.2).
3. Fit all four models using final specifications.
4. Save each model to the models/ directory.
5. Save evaluation metrics to data/model_metrics.csv.
6. Save feature metadata to models/feature_metadata.json.
"""

import argparse
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

MODELS_DIR = "models"
DATA_DIR   = "data"


def load_data(path: str) -> pd.DataFrame:
    print(f"Loading dataset from: {path}")
    df = pd.read_csv(path)
    required = ["ClaimNb", "Exposure", "DrivAge", "VehAge", "BonusMalus", "Density"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        sys.exit(f"Missing columns in dataset: {missing}")
    print(f"Dataset loaded: {len(df):,} records")
    return df


def split_data(df: pd.DataFrame):
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"Train: {len(train_df):,}  |  Test: {len(test_df):,}")
    return train_df, test_df


def fit_poisson(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Fits and evaluates the Poisson GLM but does NOT save it yet -- the
    returned model's .fittedvalues (which require the un-stripped data) are
    needed by fit_negative_binomial()'s NB2 estimation. Call save_poisson()
    after fit_negative_binomial() has consumed this model.
    """
    print("Fitting Poisson GLM...")
    model = smf.glm(
        formula="ClaimNb ~ BonusMalus + DrivAge + VehAge + Density",
        data=train_df,
        family=sm.families.Poisson(),
        offset=np.log(train_df["Exposure"]),
    ).fit()
    preds = model.predict(test_df, offset=np.log(test_df["Exposure"]))
    mae = mean_absolute_error(test_df["ClaimNb"], preds)
    aic = model.aic
    print(f"  MAE={mae:.5f}  AIC={aic:.2f}")
    return mae, aic, model


def save_poisson(model):
    if hasattr(model, "remove_data"):
        model.remove_data()
    path = os.path.join(MODELS_DIR, "poisson_model.pkl")
    model.save(path)
    print(f"  Poisson GLM saved to {path}")


def fit_negative_binomial(train_df: pd.DataFrame, test_df: pd.DataFrame, poisson_model=None):
    """Fits the Negative Binomial GLM with the dispersion parameter (alpha)
    estimated from the data via maximum likelihood, rather than fixed by
    assumption. See nb2_validation.fit_nb2_profile_mle for the full rationale
    and the profile-likelihood algorithm used (mirrors R's glm.nb).

    aic is computed with alpha counted as an estimated parameter (k=6), not
    the naive GLM-refit .aic (which would report k=5, understating AIC by 2).
    """
    from nb2_validation import fit_nb2_profile_mle, fit_poisson_glm

    print("Fitting Negative Binomial GLM (NB2, alpha estimated via profile-likelihood MLE)...")
    if poisson_model is None:
        poisson_model = fit_poisson_glm(train_df)

    nb2 = fit_nb2_profile_mle(train_df, poisson_model)
    if not nb2["converged"]:
        sys.exit(
            f"NB2 profile-likelihood estimation did not converge "
            f"(alpha={nb2['alpha']:.6f}, iterations={nb2['n_iterations']}). "
            f"Refusing to save an unstable model."
        )

    model = nb2["model"]
    alpha = nb2["alpha"]
    alpha_se = nb2["alpha_se"]
    aic = nb2["aic_correct"]
    preds = model.predict(test_df, offset=np.log(test_df["Exposure"]))
    mae = mean_absolute_error(test_df["ClaimNb"], preds)
    print(f"  alpha={alpha:.6f} (SE={alpha_se:.6f})  MAE={mae:.5f}  AIC={aic:.2f} (k=6)")
    if hasattr(model, "remove_data"):
        model.remove_data()
    path = os.path.join(MODELS_DIR, "negative_binomial_model.pkl")
    model.save(path)
    print(f"  Saved to {path}")
    return mae, aic, alpha, alpha_se


def fit_random_forest(train_df: pd.DataFrame, test_df: pd.DataFrame):
    print("Fitting Random Forest...")
    features = ["Exposure", "BonusMalus", "DrivAge", "VehAge", "Density"]
    X_train = train_df[features]
    y_train = train_df["ClaimNb"]
    X_test  = test_df[features]
    y_test  = test_df["ClaimNb"]

    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print(f"  MAE={mae:.5f}")

    importances = dict(zip(features, rf.feature_importances_))
    path = os.path.join(MODELS_DIR, "random_forest_model.pkl")
    joblib.dump(rf, path)
    print(f"  Saved to {path}")
    return mae, importances


def fit_xgboost(train_df: pd.DataFrame, test_df: pd.DataFrame):
    print("Fitting XGBoost...")
    features = ["Exposure", "BonusMalus", "DrivAge", "VehAge", "Density"]
    X_train = train_df[features]
    y_train = train_df["ClaimNb"]
    X_test  = test_df[features]
    y_test  = test_df["ClaimNb"]

    xgb = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
    )
    xgb.fit(X_train, y_train)
    preds = xgb.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print(f"  MAE={mae:.5f}")

    scores = xgb.get_booster().get_fscore()
    total  = sum(scores.values()) or 1
    importances = {k: v / total for k, v in scores.items()}

    path = os.path.join(MODELS_DIR, "xgboost_model.pkl")
    joblib.dump(xgb, path)
    print(f"  Saved to {path}")
    return mae, importances


def save_metrics(poisson_mae, poisson_aic, nb_mae, nb_aic, rf_mae, xgb_mae):
    metrics = pd.DataFrame([
        {"model": "Poisson GLM",           "model_type": "Statistical",      "mae": poisson_mae, "aic": poisson_aic, "main_strength": "Interpretability and actuarial familiarity"},
        {"model": "Negative Binomial GLM", "model_type": "Statistical",      "mae": nb_mae,      "aic": nb_aic,      "main_strength": "Accommodates overdispersion"},
        {"model": "Random Forest",         "model_type": "Machine Learning",  "mae": rf_mae,      "aic": "",          "main_strength": "Captures nonlinear effects"},
        {"model": "XGBoost",               "model_type": "Machine Learning",  "mae": xgb_mae,     "aic": "",          "main_strength": "Flexible boosted-tree prediction"},
    ])
    path = os.path.join(DATA_DIR, "model_metrics.csv")
    metrics.to_csv(path, index=False)
    print(f"Metrics saved to {path}")


def save_metadata():
    metadata = {
        "poisson":           {"features": ["BonusMalus", "DrivAge", "VehAge", "Density"], "uses_offset": True},
        "negative_binomial": {"features": ["BonusMalus", "DrivAge", "VehAge", "Density"], "uses_offset": True},
        "random_forest":     {"features": ["Exposure", "BonusMalus", "DrivAge", "VehAge", "Density"], "uses_offset": False},
        "xgboost":           {"features": ["Exposure", "BonusMalus", "DrivAge", "VehAge", "Density"], "uses_offset": False},
    }
    path = os.path.join(MODELS_DIR, "feature_metadata.json")
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Feature metadata saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="Train and save ClaimIQ models")
    parser.add_argument("--data", required=True, help="Path to freMTPL2freq.csv")
    args = parser.parse_args()

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    df = load_data(args.data)
    train_df, test_df = split_data(df)

    poisson_mae, poisson_aic, poisson_model = fit_poisson(train_df, test_df)
    nb_mae, nb_aic, nb_alpha, nb_alpha_se   = fit_negative_binomial(train_df, test_df, poisson_model=poisson_model)
    save_poisson(poisson_model)
    rf_mae, rf_importances                  = fit_random_forest(train_df, test_df)
    xgb_mae, xgb_importances                = fit_xgboost(train_df, test_df)

    save_metrics(poisson_mae, poisson_aic, nb_mae, nb_aic, rf_mae, xgb_mae)
    save_metadata()

    print("\n=== All models saved successfully ===")
    print(f"Poisson GLM  MAE={poisson_mae:.5f}  AIC={poisson_aic:.2f}")
    print(f"NB2 GLM      MAE={nb_mae:.5f}  AIC={nb_aic:.2f}  alpha={nb_alpha:.6f} (SE={nb_alpha_se:.6f})")
    print(f"Random Forest  MAE={rf_mae:.5f}")
    print(f"XGBoost        MAE={xgb_mae:.5f}")


if __name__ == "__main__":
    main()
