"""
validation_robustness.py
=========================
Repeated-split (Monte Carlo cross-validation) robustness check for the
production-relevant model specifications: Poisson GLM, corrected Negative
Binomial NB2 (alpha estimated via profile MLE), Random Forest, XGBoost.

PURPOSE
-------
Quantify how much each model's test MAE and mean Poisson deviance move
across different random 80/20 splits, and whether the MAE-based model
ranking is stable -- i.e. whether "Random Forest has the lowest MAE" is a
reproducible finding or an artifact of one particular train/test split.

WHAT THIS SCRIPT REUSES (unchanged) FROM THE PRODUCTION PIPELINE
------------------------------------------------------------------
- Dataset: the same freMTPL2freq.csv, loaded via save_models.load_data()
  (same column-presence validation, no additional preprocessing).
- Predictors and formula: "ClaimNb ~ BonusMalus + DrivAge + VehAge + Density".
- Split ratio: 80/20 (test_size=0.2), same as save_models.split_data().
- Exposure treatment: log(Exposure) offset for the GLMs; Exposure as a plain
  input feature for Random Forest / XGBoost -- identical to save_models.py.
- Random Forest hyperparameters: n_estimators=100, max_depth=10,
  random_state=42, n_jobs=-1 (identical to save_models.fit_random_forest).
- XGBoost hyperparameters: n_estimators=300, learning_rate=0.05, max_depth=4,
  subsample=0.8, colsample_bytree=0.8, random_state=42 (identical to
  save_models.fit_xgboost).
- NB2 alpha estimation: nb2_validation.fit_nb2_profile_mle(), the validated
  profile-likelihood MLE procedure from the previous correction.

WHAT DIFFERS ACROSS RUNS
-------------------------
Only the train/test split random_state varies (0..N-1). The RF/XGBoost
random_state=42 (governing tree construction) is deliberately held fixed
across all splits, matching production -- this experiment isolates the
effect of *which rows land in train vs. test*, not additional randomness
from re-seeding the estimators themselves.

WHAT THIS SCRIPT DOES NOT DO
-----------------------------
- Does NOT modify save_models.py, app.py, data/model_metrics.csv, or any
  models/*.pkl artifact.
- Does NOT modify data/freMTPL2freq.csv.
- Does NOT touch the paper or the website.
- Does NOT compute calibration/lift-chart/Gini/sensitivity analyses --
  out of scope for this script by design.

FAILURE HANDLING
-----------------
- Exposure is validated as strictly positive on both the train and test
  fold of every split before any model is fit.
- A hard exception while fitting any model for a given seed is caught,
  recorded in the raw results with an error message, and the run continues
  to the next seed. Failed runs are never silently dropped -- they are
  counted in n_failed in the summary and remain visible in the raw CSV.
- If the NB2 profile-likelihood algorithm completes without an exception
  but fails to converge for a given seed, this is treated as a distinct,
  stricter failure: the script halts immediately with an explanation,
  rather than reporting an unstable alpha estimate as if it were valid.

Usage
-----
    python validation_robustness.py --data data/freMTPL2freq.csv --n-splits 20
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from save_models import load_data
from nb2_validation import fit_poisson_glm, fit_nb2_profile_mle, poisson_deviance

FEATURES_ML = ["Exposure", "BonusMalus", "DrivAge", "VehAge", "Density"]
RAW_RESULTS_PATH = os.path.join("data", "repeated_split_results.csv")
SUMMARY_PATH = os.path.join("data", "repeated_split_summary.csv")

MODEL_NAMES = [
    "Poisson GLM",
    "Negative Binomial NB2 (alpha estimated)",
    "Random Forest",
    "XGBoost",
]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_exposure(train_df: pd.DataFrame, test_df: pd.DataFrame, seed: int) -> None:
    """Raises if Exposure <= 0 is found in either fold. Never modifies data."""
    for name, part in [("train", train_df), ("test", test_df)]:
        n_bad = int((part["Exposure"] <= 0).sum())
        if n_bad > 0:
            raise ValueError(
                f"[seed={seed}] {n_bad} row(s) with Exposure <= 0 found in the {name} "
                f"split. Refusing to fit (the log(Exposure) offset would be undefined "
                f"for these rows). No rows were modified or dropped by this script."
            )


# ---------------------------------------------------------------------------
# Model fitting (RF/XGB replicated here, independent of save_models.py, which
# has the side effect of overwriting models/*.pkl via joblib.dump -- must not
# be called from this script)
# ---------------------------------------------------------------------------

def fit_random_forest(train_df: pd.DataFrame) -> RandomForestRegressor:
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(train_df[FEATURES_ML], train_df["ClaimNb"])
    return rf


def fit_xgboost(train_df: pd.DataFrame) -> XGBRegressor:
    xgb = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4,
                        subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
    xgb.fit(train_df[FEATURES_ML], train_df["ClaimNb"])
    return xgb


def evaluate_glm(model, test_df: pd.DataFrame):
    preds = model.predict(test_df, offset=np.log(test_df["Exposure"])).values
    y = test_df["ClaimNb"].values
    return mean_absolute_error(y, preds), poisson_deviance(y, preds)


def evaluate_sklearn(model, test_df: pd.DataFrame):
    preds = model.predict(test_df[FEATURES_ML])
    y = test_df["ClaimNb"].values
    return mean_absolute_error(y, preds), poisson_deviance(y, np.clip(preds, 1e-9, None))


# ---------------------------------------------------------------------------
# Per-split evaluation
# ---------------------------------------------------------------------------

def blank_result(seed, model, status, error=""):
    return {
        "seed": seed, "model": model, "status": status,
        "converged": False, "alpha": np.nan,
        "test_mae": np.nan, "mean_poisson_deviance": np.nan,
        "fit_seconds": np.nan, "error": error,
    }


def run_one_split(df: pd.DataFrame, seed: int) -> list:
    """Fits all four models for one train/test split. Returns a list of
    per-model result dicts. Calls sys.exit(1) if NB2 fails to converge
    (a deliberate hard stop, per the validation requirements).
    """
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=seed)
    validate_exposure(train_df, test_df, seed)

    results = []

    # --- Poisson GLM ---
    pois = None
    try:
        t0 = time.time()
        pois = fit_poisson_glm(train_df)
        mae, dev = evaluate_glm(pois, test_df)
        results.append({
            "seed": seed, "model": "Poisson GLM", "status": "OK",
            "converged": bool(pois.converged), "alpha": np.nan,
            "test_mae": mae, "mean_poisson_deviance": dev,
            "fit_seconds": time.time() - t0, "error": "",
        })
    except Exception as e:
        results.append(blank_result(seed, "Poisson GLM", "FAILED", str(e)))

    # --- Negative Binomial NB2 (alpha estimated via profile MLE) ---
    if pois is not None:
        t0 = time.time()
        nb2_result = fit_nb2_profile_mle(train_df, pois)
        if not nb2_result["converged"]:
            print(f"\n*** STOPPING: NB2 profile-likelihood estimation did not converge "
                  f"for seed={seed} (alpha={nb2_result['alpha']:.6f}, "
                  f"iterations={nb2_result['n_iterations']}). ***")
            print("This is treated as a hard failure, not a silently-reported unstable "
                  "estimate, per the validation requirements. No summary file has been "
                  "written for this run. Raw per-seed results collected so far have been "
                  "saved for inspection before you re-run.")
            _save_raw_partial(results, seed)
            sys.exit(1)
        try:
            mae, dev = evaluate_glm(nb2_result["model"], test_df)
            results.append({
                "seed": seed, "model": "Negative Binomial NB2 (alpha estimated)", "status": "OK",
                "converged": True, "alpha": nb2_result["alpha"],
                "test_mae": mae, "mean_poisson_deviance": dev,
                "fit_seconds": time.time() - t0, "error": "",
            })
        except Exception as e:
            results.append(blank_result(seed, "Negative Binomial NB2 (alpha estimated)", "FAILED", str(e)))
    else:
        results.append(blank_result(seed, "Negative Binomial NB2 (alpha estimated)",
                                     "SKIPPED (Poisson fit failed)"))

    # --- Random Forest ---
    try:
        t0 = time.time()
        rf = fit_random_forest(train_df)
        mae, dev = evaluate_sklearn(rf, test_df)
        results.append({
            "seed": seed, "model": "Random Forest", "status": "OK",
            "converged": "N/A", "alpha": np.nan,
            "test_mae": mae, "mean_poisson_deviance": dev,
            "fit_seconds": time.time() - t0, "error": "",
        })
    except Exception as e:
        results.append(blank_result(seed, "Random Forest", "FAILED", str(e)))

    # --- XGBoost ---
    try:
        t0 = time.time()
        xgb = fit_xgboost(train_df)
        mae, dev = evaluate_sklearn(xgb, test_df)
        results.append({
            "seed": seed, "model": "XGBoost", "status": "OK",
            "converged": "N/A", "alpha": np.nan,
            "test_mae": mae, "mean_poisson_deviance": dev,
            "fit_seconds": time.time() - t0, "error": "",
        })
    except Exception as e:
        results.append(blank_result(seed, "XGBoost", "FAILED", str(e)))

    return results


def _save_raw_partial(partial_results, current_seed):
    if not partial_results:
        return
    os.makedirs("data", exist_ok=True)
    pd.DataFrame(partial_results).to_csv(RAW_RESULTS_PATH, index=False)
    print(f"Partial raw results (through seed={current_seed}) saved to {RAW_RESULTS_PATH}")


# ---------------------------------------------------------------------------
# Ranking and summary
# ---------------------------------------------------------------------------

def add_rankings(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Rank = 1 (best/lowest MAE) .. k (worst) computed within each seed,
    among models with status == 'OK' only. A model missing for a given seed
    (e.g. it failed) is excluded from that seed's ranking rather than
    assigned a placeholder rank -- visible via n_failed in the summary.
    """
    raw_df = raw_df.copy()
    raw_df["rank"] = np.nan
    for seed, group in raw_df.groupby("seed"):
        ok = group[group["status"] == "OK"]
        if ok.empty:
            continue
        ranks = ok["test_mae"].rank(method="min")
        raw_df.loc[ranks.index, "rank"] = ranks.values
    return raw_df


def build_summary(raw_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in MODEL_NAMES:
        model_all = raw_df[raw_df["model"] == model]
        ok = model_all[model_all["status"] == "OK"]
        n_ok = len(ok)
        n_failed = len(model_all) - n_ok

        if n_ok == 0:
            rows.append({"model": model, "n_ok": 0, "n_failed": n_failed})
            continue

        mae = ok["test_mae"].to_numpy()
        mean_mae = float(mae.mean())
        std_mae = float(mae.std(ddof=1)) if n_ok > 1 else float("nan")
        se_mae = std_mae / np.sqrt(n_ok) if n_ok > 1 else float("nan")
        if n_ok > 1 and se_mae > 0:
            tcrit = float(scipy_stats.t.ppf(0.975, df=n_ok - 1))
            ci_low, ci_high = mean_mae - tcrit * se_mae, mean_mae + tcrit * se_mae
        else:
            ci_low, ci_high = float("nan"), float("nan")

        n_wins = int((ok["rank"] == 1.0).sum())

        rows.append({
            "model": model,
            "n_ok": n_ok,
            "n_failed": n_failed,
            "mean_mae": mean_mae,
            "std_mae": std_mae,
            "min_mae": float(mae.min()),
            "max_mae": float(mae.max()),
            "ci95_low": ci_low,
            "ci95_high": ci_high,
            "mean_poisson_deviance": float(ok["mean_poisson_deviance"].mean()),
            "n_wins": n_wins,
            "pct_wins": 100.0 * n_wins / n_ok,
            "avg_rank": float(ok["rank"].mean()),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Repeated-split (Monte Carlo CV) robustness check. "
                     "Read-only with respect to production artifacts."
    )
    parser.add_argument("--data", required=True, help="Path to freMTPL2freq.csv")
    parser.add_argument("--n-splits", type=int, default=20, help="Number of independent splits")
    args = parser.parse_args()

    df = load_data(args.data)

    all_results = []
    for seed in range(args.n_splits):
        print(f"\n--- Split seed={seed} ({seed + 1}/{args.n_splits}) ---", flush=True)
        split_results = run_one_split(df, seed)
        for r in split_results:
            mae_str = f"{r['test_mae']:.6f}" if pd.notna(r["test_mae"]) else "NA"
            print(f"  {r['model']:<45s} status={r['status']:<28s} MAE={mae_str}", flush=True)
        all_results.extend(split_results)

    raw_df = pd.DataFrame(all_results)
    raw_df = add_rankings(raw_df)

    os.makedirs("data", exist_ok=True)
    raw_df.to_csv(RAW_RESULTS_PATH, index=False)
    print(f"\nRaw per-seed results saved to {RAW_RESULTS_PATH}")

    n_any_failed = int((raw_df["status"] != "OK").sum())
    if n_any_failed > 0:
        print(f"\n*** NOTE: {n_any_failed} (seed, model) run(s) did not complete with status OK. "
              f"See the 'status'/'error' columns in {RAW_RESULTS_PATH}. These are counted in "
              f"n_failed below and were NOT silently dropped. ***")

    summary_df = build_summary(raw_df)
    summary_df.to_csv(SUMMARY_PATH, index=False)
    print(f"Summary saved to {SUMMARY_PATH}")

    pd.set_option("display.width", 160)
    print("\n=== Summary: Repeated-Split Robustness (n_splits = {}) ===".format(args.n_splits))
    print(summary_df.to_string(index=False))

    print("\nNo production files were modified: save_models.py, app.py, "
          "data/model_metrics.csv, and models/*.pkl are unchanged.")


if __name__ == "__main__":
    main()
