"""
generate_research_results.py
=============================
Builds data/research_results.json, the single source of truth the ClaimIQ
website reads for every displayed model metric, validation result, and
feature importance. Run this after save_models.py (which produces the
production models/*.pkl and data/model_metrics.csv) whenever those change.

Sources (see README section of this docstring's module-level SOURCES dict
for the precise mapping used by each field):
- data/freMTPL2freq.csv          (dataset size, zero-claim proportion)
- data/model_metrics.csv         (primary-split MAE/AIC/main_strength)
- data/nb_comparison_report.csv  (log-likelihood, k_params, coefficients, alpha/alpha_se)
- data/repeated_split_summary.csv (20-split validation: mean/std/CI/wins/rank)
- models/random_forest_model.pkl, models/xgboost_model.pkl, models/feature_metadata.json
  (feature importances, extracted LIVE from the fitted objects -- never hand-copied)

Usage
-----
    python generate_research_results.py
"""

import ast
import json
import re
from datetime import datetime, timezone

import joblib
import pandas as pd

DATA_DIR = "data"
MODELS_DIR = "models"
OUT_PATH = f"{DATA_DIR}/research_results.json"

SOURCES = {
    "model_metrics": f"{DATA_DIR}/model_metrics.csv",
    "nb_comparison": f"{DATA_DIR}/nb_comparison_report.csv",
    "repeated_split_summary": f"{DATA_DIR}/repeated_split_summary.csv",
    "repeated_split_raw": f"{DATA_DIR}/repeated_split_results.csv",
    "feature_metadata": f"{MODELS_DIR}/feature_metadata.json",
    "dataset": f"{DATA_DIR}/freMTPL2freq.csv",
}


def parse_coefficients(cell: str) -> dict:
    """Parses a stringified Python dict like "{'Intercept': np.float64(-3.73...), ...}"
    safely (no eval()) by stripping the np.float64(...) wrapper before ast.literal_eval.
    """
    cleaned = re.sub(r"np\.float64\(([^)]+)\)", r"\1", cell)
    return {k: float(v) for k, v in ast.literal_eval(cleaned).items()}


def load_dataset_stats() -> dict:
    df = pd.read_csv(SOURCES["dataset"], usecols=["ClaimNb", "Exposure"])
    return {
        "name": "French MTPL2freq",
        "n_records": int(len(df)),
        "train_test_split": {"test_size": 0.2, "random_state": 42},
        "pct_zero_claims": round(float((df["ClaimNb"] == 0).mean()) * 100, 2),
        "variables": [
            {"name": "ClaimNb", "role": "response", "description": "Number of claims reported"},
            {"name": "Exposure", "role": "offset", "description": "Policy exposure (fraction of year)"},
            {"name": "DrivAge", "role": "predictor", "description": "Age of insured driver"},
            {"name": "VehAge", "role": "predictor", "description": "Age of insured vehicle"},
            {"name": "BonusMalus", "role": "predictor", "description": "Claims-history experience score"},
            {"name": "Density", "role": "predictor", "description": "Population density of policyholder's area"},
        ],
    }


def load_metrics() -> pd.DataFrame:
    return pd.read_csv(SOURCES["model_metrics"])


def load_nb_comparison() -> pd.DataFrame:
    df = pd.read_csv(SOURCES["nb_comparison"])
    df["coefficients"] = df["coefficients"].apply(parse_coefficients)
    return df


def load_repeated_split_summary() -> pd.DataFrame:
    return pd.read_csv(SOURCES["repeated_split_summary"])


def live_feature_importance(model_key: str, pkl_name: str) -> dict:
    """Extracts feature importance directly from the fitted model object at
    generation time, matching the exact logic already used in save_models.py
    (rf.feature_importances_ / xgb.get_booster().get_fscore(), normalized).
    Never copied from a hardcoded snapshot -- self-corrects if the model is retrained.
    """
    with open(SOURCES["feature_metadata"]) as f:
        meta = json.load(f)
    features = meta[model_key]["features"]
    model = joblib.load(f"{MODELS_DIR}/{pkl_name}")

    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
        importance = dict(zip(features, [float(v) for v in values]))
    else:
        scores = model.get_booster().get_fscore()
        total = sum(scores.values()) or 1
        importance = {k: float(v) / total for k, v in scores.items()}
        importance = {f: importance.get(f, 0.0) for f in features}

    return dict(sorted(importance.items(), key=lambda kv: -kv[1]))


def build():
    metrics = load_metrics().set_index("model")
    nb_cmp = load_nb_comparison().set_index("model")
    rs = load_repeated_split_summary().set_index("model")

    poisson_row = metrics.loc["Poisson GLM"]
    poisson_cmp = nb_cmp.loc["Poisson GLM"]
    poisson_rs = rs.loc["Poisson GLM"]

    nb_row = metrics.loc["Negative Binomial GLM"]
    nb2_cmp = nb_cmp.loc["Negative Binomial NB2 (alpha estimated via profile MLE)"]
    nb1_cmp = nb_cmp.loc["Negative Binomial GLM (alpha=1, fixed)"]
    nb_rs = rs.loc["Negative Binomial NB2 (alpha estimated)"]

    rf_row = metrics.loc["Random Forest"]
    rf_rs = rs.loc["Random Forest"]

    xgb_row = metrics.loc["XGBoost"]
    xgb_rs = rs.loc["XGBoost"]

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generated_by": "generate_research_results.py",
        "source_files": SOURCES,
        "dataset": load_dataset_stats(),
        "models": {
            "poisson": {
                "display_name": "Poisson GLM",
                "model_type": "Statistical",
                "primary_split": {
                    "mae": round(float(poisson_row["mae"]), 5),
                    "aic": round(float(poisson_row["aic"]), 2),
                    "log_likelihood": round(float(poisson_cmp["log_likelihood"]), 2),
                    "k_params": int(poisson_cmp["k_params"]),
                    "converged": bool(poisson_cmp["converged"]),
                    "coefficients": poisson_cmp["coefficients"],
                },
                "repeated_split": {
                    "n_splits": int(poisson_rs["n_ok"]),
                    "mean_mae": round(float(poisson_rs["mean_mae"]), 6),
                    "std_mae": round(float(poisson_rs["std_mae"]), 6),
                    "ci95_low": round(float(poisson_rs["ci95_low"]), 6),
                    "ci95_high": round(float(poisson_rs["ci95_high"]), 6),
                    "mean_poisson_deviance": round(float(poisson_rs["mean_poisson_deviance"]), 6),
                    "n_wins": int(poisson_rs["n_wins"]),
                    "pct_wins": float(poisson_rs["pct_wins"]),
                    "avg_rank": float(poisson_rs["avg_rank"]),
                },
                "main_strength": poisson_row["main_strength"],
            },
            "negative_binomial": {
                "display_name": "Negative Binomial GLM (NB2, alpha estimated)",
                "model_type": "Statistical",
                "primary_split": {
                    "mae": round(float(nb_row["mae"]), 5),
                    "aic": round(float(nb_row["aic"]), 2),
                    "log_likelihood": round(float(nb2_cmp["log_likelihood"]), 2),
                    "k_params": int(nb2_cmp["k_params"]),
                    "converged": bool(nb2_cmp["converged"]),
                    "alpha": round(float(nb2_cmp["alpha"]), 5),
                    "alpha_se": round(float(nb2_cmp["alpha_se"]), 5),
                    "coefficients": nb2_cmp["coefficients"],
                },
                "repeated_split": {
                    "n_splits": int(nb_rs["n_ok"]),
                    "mean_mae": round(float(nb_rs["mean_mae"]), 6),
                    "std_mae": round(float(nb_rs["std_mae"]), 6),
                    "ci95_low": round(float(nb_rs["ci95_low"]), 6),
                    "ci95_high": round(float(nb_rs["ci95_high"]), 6),
                    "mean_poisson_deviance": round(float(nb_rs["mean_poisson_deviance"]), 6),
                    "n_wins": int(nb_rs["n_wins"]),
                    "pct_wins": float(nb_rs["pct_wins"]),
                    "avg_rank": float(nb_rs["avg_rank"]),
                },
                "prior_fixed_alpha_comparison": {
                    "alpha": 1.0,
                    "mae": round(float(nb1_cmp["test_mae"]), 5),
                    "aic": round(float(nb1_cmp["aic"]), 2),
                    "k_params": int(nb1_cmp["k_params"]),
                    "note": "Superseded production specification. alpha was assumed rather than "
                            "estimated, and AIC understated by 2 points from not counting alpha "
                            "as a free parameter (k=5 instead of k=6).",
                },
                "main_strength": nb_row["main_strength"],
            },
            "random_forest": {
                "display_name": "Random Forest",
                "model_type": "Machine Learning",
                "primary_split": {"mae": round(float(rf_row["mae"]), 5), "aic": None},
                "repeated_split": {
                    "n_splits": int(rf_rs["n_ok"]),
                    "mean_mae": round(float(rf_rs["mean_mae"]), 6),
                    "std_mae": round(float(rf_rs["std_mae"]), 6),
                    "ci95_low": round(float(rf_rs["ci95_low"]), 6),
                    "ci95_high": round(float(rf_rs["ci95_high"]), 6),
                    "mean_poisson_deviance": round(float(rf_rs["mean_poisson_deviance"]), 6),
                    "n_wins": int(rf_rs["n_wins"]),
                    "pct_wins": float(rf_rs["pct_wins"]),
                    "avg_rank": float(rf_rs["avg_rank"]),
                },
                "feature_importance": live_feature_importance("random_forest", "random_forest_model.pkl"),
                "main_strength": rf_row["main_strength"],
            },
            "xgboost": {
                "display_name": "XGBoost",
                "model_type": "Machine Learning",
                "primary_split": {"mae": round(float(xgb_row["mae"]), 5), "aic": None},
                "repeated_split": {
                    "n_splits": int(xgb_rs["n_ok"]),
                    "mean_mae": round(float(xgb_rs["mean_mae"]), 6),
                    "std_mae": round(float(xgb_rs["std_mae"]), 6),
                    "ci95_low": round(float(xgb_rs["ci95_low"]), 6),
                    "ci95_high": round(float(xgb_rs["ci95_high"]), 6),
                    "mean_poisson_deviance": round(float(xgb_rs["mean_poisson_deviance"]), 6),
                    "n_wins": int(xgb_rs["n_wins"]),
                    "pct_wins": float(xgb_rs["pct_wins"]),
                    "avg_rank": float(xgb_rs["avg_rank"]),
                },
                "feature_importance": live_feature_importance("xgboost", "xgboost_model.pkl"),
                "main_strength": xgb_row["main_strength"],
            },
        },
    }

    aic_delta = result["models"]["negative_binomial"]["primary_split"]["aic"] - result["models"]["poisson"]["primary_split"]["aic"]
    result["cross_model"] = {
        "top_feature_all_models": "BonusMalus",
        "aic_delta_nb2_vs_poisson": round(aic_delta, 2),
        "repeated_split_best_avg_rank": "random_forest",
        "predictive_ceiling_note": (
            "MAE spread across all 4 models is under 0.002 on both the primary split and "
            "across 20 repeated splits, though the tree-based models' advantage over both "
            "GLMs is statistically reproducible (non-overlapping 95% CIs), not noise."
        ),
    }
    return result


if __name__ == "__main__":
    data = build()
    with open(OUT_PATH, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {OUT_PATH}")
    print(f"  Poisson:  MAE={data['models']['poisson']['primary_split']['mae']}  AIC={data['models']['poisson']['primary_split']['aic']}")
    nb = data["models"]["negative_binomial"]
    print(f"  NB2:      MAE={nb['primary_split']['mae']}  AIC={nb['primary_split']['aic']}  alpha={nb['primary_split']['alpha']}")
    print(f"  RF:       MAE={data['models']['random_forest']['primary_split']['mae']}  "
          f"repeated-split wins={data['models']['random_forest']['repeated_split']['n_wins']}/20")
    print(f"  XGBoost:  MAE={data['models']['xgboost']['primary_split']['mae']}  "
          f"repeated-split wins={data['models']['xgboost']['repeated_split']['n_wins']}/20")
    print(f"  AIC delta (NB2 vs Poisson): {data['cross_model']['aic_delta_nb2_vs_poisson']}")
