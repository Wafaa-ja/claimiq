"""
nb2_validation.py
=================
Validates the Negative Binomial GLM specification used in ClaimIQ and compares
it against a properly MLE-estimated NB2 alternative.

WHAT THIS SCRIPT DOES
----------------------
1. Loads freMTPL2freq.csv and reports data-quality checks (missing values,
   non-positive Exposure, duplicate rows, extreme values). Report-only: no row
   is ever dropped, capped, or modified.
2. Reproduces the exact production train/test split (test_size=0.2,
   random_state=42) by importing load_data()/split_data() from save_models.py.
3. Fits three models on that split:
     - Poisson GLM (unchanged specification)
     - Negative Binomial GLM with alpha fixed at 1 (the CURRENT production
       specification -- kept as-is for comparison)
     - Negative Binomial NB2 with alpha estimated from the data via maximum
       likelihood, using a profile-likelihood alternating algorithm (see
       fit_nb2_profile_mle for why this algorithm is used instead of
       statsmodels' joint discrete MLE estimator).
4. Reports, for each model: convergence status, coefficients, alpha (where
   applicable) with its standard error, log-likelihood, number of estimated
   parameters, AIC, test-set MAE, and test-set mean Poisson deviance.
5. Writes the comparison to data/nb_comparison_report.csv (a NEW file).

WHAT THIS SCRIPT DOES NOT DO
-----------------------------
- Does NOT modify data/freMTPL2freq.csv.
- Does NOT overwrite models/poisson_model.pkl, models/negative_binomial_model.pkl,
  or any Random Forest / XGBoost artifact.
- Does NOT overwrite data/model_metrics.csv.
- Does NOT touch app.py, utils/model_loader.py, or utils/prediction.py.

Usage
-----
    python nb2_validation.py --data data/freMTPL2freq.csv
"""

import argparse
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.optimize import minimize_scalar
from scipy.special import gammaln
from sklearn.metrics import mean_absolute_error

from save_models import load_data, split_data

FORMULA = "ClaimNb ~ BonusMalus + DrivAge + VehAge + Density"
REPORT_PATH = os.path.join("data", "nb_comparison_report.csv")


# ---------------------------------------------------------------------------
# Data validation (report-only -- never drops or modifies rows)
# ---------------------------------------------------------------------------

def report_data_quality(df: pd.DataFrame) -> None:
    """Prints a data-quality report. Does not mutate df or return a cleaned copy."""
    print("\n=== Data Quality Report (read-only -- no rows modified) ===")
    required = ["ClaimNb", "Exposure", "DrivAge", "VehAge", "BonusMalus", "Density"]

    missing = df[required].isna().sum()
    print("\nMissing values:")
    print(missing.to_string())

    n_nonpos_exposure = int((df["Exposure"] <= 0).sum())
    print(f"\nExposure <= 0: {n_nonpos_exposure}  "
          f"(min={df['Exposure'].min():.6g}, max={df['Exposure'].max():.6g})")

    n_dupe_rows = int(df.duplicated().sum())
    print(f"Full-row duplicates: {n_dupe_rows}")
    if "IDpol" in df.columns:
        print(f"Duplicate IDpol: {int(df['IDpol'].duplicated().sum())}")

    print("\nExtreme-value summary:")
    for col in ["VehAge", "DrivAge", "BonusMalus", "Density"]:
        print(f"  {col}: min={df[col].min()}  max={df[col].max()}  "
              f"p99={df[col].quantile(0.99):.2f}  p99.9={df[col].quantile(0.999):.2f}")

    print(f"\nVehAge == 100 (suspected placeholder cluster): "
          f"{int((df['VehAge'] == 100).sum())} rows")
    print(f"ClaimNb >= 8: {int((df['ClaimNb'] >= 8).sum())} rows (max={df['ClaimNb'].max()})")

    implied_rate = df["ClaimNb"] / df["Exposure"]
    print(f"Rows with implied claim rate (ClaimNb/Exposure) > 30: "
          f"{int((implied_rate > 30).sum())} (max={implied_rate.max():.1f})")

    if n_nonpos_exposure > 0 or n_dupe_rows > 0 or int(missing.sum()) > 0:
        print("\n*** ACTION NEEDED: blocking data-quality issues detected above. "
              "No rows have been removed or modified. Review before proceeding. ***")
    else:
        print("\nNo blocking issues found (no missing values, no non-positive "
              "Exposure, no duplicate rows). The extreme-value rows noted above "
              "are real, present-in-data values and have been left untouched "
              "pending an explicit decision on whether to add a separate, "
              "clearly-labeled sensitivity run.")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def poisson_deviance(y_true: np.ndarray, mu_pred: np.ndarray) -> float:
    """Mean Poisson deviance: 2*(y*log(y/mu) - (y-mu)).

    The y=0 term is defined as 0 by the standard convention (lim_{y->0} y*log(y) = 0).
    mu is floored at a small epsilon only to avoid a division-by-zero on a
    degenerate prediction; this does not alter any fitted model or the data.
    """
    mu = np.clip(np.asarray(mu_pred, dtype=float), 1e-9, None)
    y = np.asarray(y_true, dtype=float)
    term = np.where(y > 0, y * np.log(y / mu), 0.0)
    dev = 2.0 * (term - (y - mu))
    return float(np.mean(dev))


def evaluate_on_test(model, test_df: pd.DataFrame) -> dict:
    preds = model.predict(test_df, offset=np.log(test_df["Exposure"]))
    mae = mean_absolute_error(test_df["ClaimNb"], preds)
    dev = poisson_deviance(test_df["ClaimNb"].values, preds.values)
    return {"mae": mae, "mean_poisson_deviance": dev}


# ---------------------------------------------------------------------------
# Model fitting
#
# NOTE: we deliberately do NOT call save_models.fit_poisson()/fit_negative_binomial()
# here -- those functions have the side effect of overwriting models/poisson_model.pkl
# and models/negative_binomial_model.pkl via model.save(path). This script must not
# touch production artifacts, so the same formula/family/offset specification is
# replicated independently below, without any save step.
# ---------------------------------------------------------------------------

def fit_poisson_glm(train_df: pd.DataFrame):
    return smf.glm(
        formula=FORMULA, data=train_df,
        family=sm.families.Poisson(),
        offset=np.log(train_df["Exposure"]),
    ).fit()


def fit_nb_fixed_alpha(train_df: pd.DataFrame, alpha: float = 1.0):
    return smf.glm(
        formula=FORMULA, data=train_df,
        family=sm.families.NegativeBinomial(alpha=alpha),
        offset=np.log(train_df["Exposure"]),
    ).fit()


def _nb2_neg_loglik_given_mu(alpha: float, y: np.ndarray, mu: np.ndarray) -> float:
    """Negative NB2 log-likelihood as a function of alpha alone, holding mu fixed.
    This is the profile log-likelihood maximized at each step of the alternating
    algorithm in fit_nb2_profile_mle.
    """
    r = 1.0 / alpha
    ll = (gammaln(y + r) - gammaln(r) - gammaln(y + 1)
          + r * np.log(r / (r + mu)) + y * np.log(mu / (r + mu)))
    return -np.sum(ll)


def fit_nb2_profile_mle(train_df: pd.DataFrame, poisson_model, max_iter: int = 15, tol: float = 1e-6):
    """Estimate the NB2 dispersion parameter alpha via maximum likelihood, using an
    alternating profile-likelihood algorithm -- the same approach underlying R's
    canonical MASS::glm.nb:

        repeat:
            (1) with mu fixed, solve a 1-D bounded MLE for alpha
            (2) with alpha fixed, refit the GLM (IRLS) to update mu
        until alpha stabilizes

    WHY THIS ALGORITHM (not statsmodels' joint discrete MLE):
    statsmodels' joint (beta, alpha) MLE estimator (smf.negativebinomial /
    sm.NegativeBinomial) was tested against this exact dataset/split under
    multiple configurations -- default start values, Poisson-informed start
    values, method='bfgs'/'newton'/'nm'/'lbfgs'/'powell', standardized
    predictors, and literature-standard outlier capping (ClaimNb<=4,
    Exposure<=1) -- and failed to converge cleanly in every case (alpha
    diverging to infinity or the log-likelihood becoming NaN from overflow in
    the joint parameter search). The 1-D profile step below only ever
    optimizes a single parameter at a time, which avoids the joint numerical
    instability, while still producing a genuine maximum-likelihood estimate
    of alpha (this is a standard, citable alternative algorithm for NB2 MLE,
    not an ad hoc shortcut).

    NOTE on tol/max_iter: empirically (checked across multiple splits), the
    profile negative log-likelihood becomes numerically flat and alpha
    stabilizes to 5+ decimal places within 3-4 iterations. Consecutive-
    iteration deltas in alpha then oscillate in the 1e-7..1e-8 range purely
    from floating-point noise in the repeated GLM refits -- an overly strict
    tolerance (e.g. 1e-8) can therefore fail to register convergence for many
    extra iterations even though the estimate is already stable, which is a
    tolerance-selection issue, not a sign of genuine non-convergence. tol=1e-6
    is comfortably tighter than any precision that matters given alpha's
    typical standard error (~0.05 in this dataset), and max_iter=15 gives a
    safety margin beyond the ~4-6 iterations actually needed.
    """
    y = train_df["ClaimNb"].values
    mu = poisson_model.fittedvalues.values
    alpha = 1.0
    model = poisson_model  # placeholder until the first NB refit below
    converged = False
    n_iter = max_iter

    for i in range(max_iter):
        res = minimize_scalar(
            _nb2_neg_loglik_given_mu, args=(y, mu),
            bounds=(1e-6, 20.0), method="bounded",
            options={"xatol": 1e-10},
        )
        alpha_new = float(res.x)
        model = fit_nb_fixed_alpha(train_df, alpha=alpha_new)
        mu_new = model.fittedvalues.values

        delta = abs(alpha_new - alpha)
        alpha = alpha_new
        mu = mu_new

        if delta < tol:
            converged = True
            n_iter = i + 1
            break

    # Standard error of alpha via the numerical second derivative (Hessian) of
    # the profile negative log-likelihood at the optimum.
    eps = 1e-4
    f = lambda a: _nb2_neg_loglik_given_mu(a, y, mu)
    d2 = (f(alpha + eps) - 2.0 * f(alpha) + f(alpha - eps)) / eps ** 2
    alpha_se = float(np.sqrt(1.0 / d2)) if d2 > 0 else float("nan")

    joint_loglik = -_nb2_neg_loglik_given_mu(alpha, y, mu)
    k_params = len(model.params) + 1  # +1: alpha is now genuinely estimated, not given
    aic_correct = -2.0 * joint_loglik + 2.0 * k_params

    return {
        # Same object type (GLMResultsWrapper) as the current production NB model,
        # so it remains compatible with sm.load()/predict_glm() unchanged, should
        # it later be promoted to production.
        "model": model,
        "alpha": alpha,
        "alpha_se": alpha_se,
        "joint_loglik": joint_loglik,
        "aic_correct": aic_correct,
        "k_params": k_params,
        "n_iterations": n_iter,
        "converged": converged,
    }


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def build_comparison_table(poisson_model, nb_fixed_model, nb2_result, test_df) -> pd.DataFrame:
    rows = []

    ev = evaluate_on_test(poisson_model, test_df)
    rows.append({
        "model": "Poisson GLM",
        "converged": bool(poisson_model.converged),
        "alpha": "",
        "alpha_se": "",
        "log_likelihood": poisson_model.llf,
        "k_params": len(poisson_model.params),
        "aic": poisson_model.aic,
        "test_mae": ev["mae"],
        "mean_poisson_deviance": ev["mean_poisson_deviance"],
        "coefficients": dict(poisson_model.params),
    })

    ev = evaluate_on_test(nb_fixed_model, test_df)
    rows.append({
        "model": "Negative Binomial GLM (alpha=1, fixed)",
        "converged": bool(nb_fixed_model.converged),
        "alpha": 1.0,
        "alpha_se": "not estimated (fixed by assumption)",
        "log_likelihood": nb_fixed_model.llf,
        # alpha is NOT counted as a free parameter here: it was supplied, not
        # estimated, so k stays at 5 -- this matches how statsmodels' own .aic
        # already computes it, and matches the original paper's reported figure.
        "k_params": len(nb_fixed_model.params),
        "aic": nb_fixed_model.aic,
        "test_mae": ev["mae"],
        "mean_poisson_deviance": ev["mean_poisson_deviance"],
        "coefficients": dict(nb_fixed_model.params),
    })

    nb2_model = nb2_result["model"]
    ev = evaluate_on_test(nb2_model, test_df)
    rows.append({
        "model": "Negative Binomial NB2 (alpha estimated via profile MLE)",
        "converged": nb2_result["converged"],
        "alpha": nb2_result["alpha"],
        "alpha_se": nb2_result["alpha_se"],
        # Joint log-likelihood/AIC computed manually (see fit_nb2_profile_mle):
        # k=6 because alpha is now an estimated parameter and must count as one.
        # Do NOT read nb2_model.aic directly here -- that GLM-refit-at-fixed-alpha
        # object reports k=5 (it has no way to know alpha was estimated
        # elsewhere), which would understate the true AIC by exactly 2.
        "log_likelihood": nb2_result["joint_loglik"],
        "k_params": nb2_result["k_params"],
        "aic": nb2_result["aic_correct"],
        "test_mae": ev["mae"],
        "mean_poisson_deviance": ev["mean_poisson_deviance"],
        "coefficients": dict(nb2_model.params),
    })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Validate and compare Negative Binomial specifications. "
                     "Read-only with respect to production artifacts."
    )
    parser.add_argument("--data", required=True, help="Path to freMTPL2freq.csv")
    args = parser.parse_args()

    df = load_data(args.data)
    report_data_quality(df)

    # Identical split to production: test_size=0.2, random_state=42 (see save_models.split_data)
    train_df, test_df = split_data(df)

    print("\nFitting Poisson GLM (baseline, unchanged specification)...")
    poisson_model = fit_poisson_glm(train_df)

    print("Fitting Negative Binomial GLM (alpha=1, fixed -- current production specification)...")
    nb_fixed_model = fit_nb_fixed_alpha(train_df, alpha=1.0)

    print("Fitting Negative Binomial NB2 (alpha estimated via profile-likelihood MLE)...")
    nb2_result = fit_nb2_profile_mle(train_df, poisson_model)
    print(f"  alpha_hat={nb2_result['alpha']:.6f}  se={nb2_result['alpha_se']:.6f}  "
          f"converged={nb2_result['converged']}  iterations={nb2_result['n_iterations']}")

    table = build_comparison_table(poisson_model, nb_fixed_model, nb2_result, test_df)

    pd.set_option("display.width", 160)
    pd.set_option("display.max_colwidth", 80)
    print("\n=== Model Comparison ===")
    print(table.drop(columns=["coefficients"]).to_string(index=False))

    print("\n=== Coefficients per model ===")
    for _, row in table.iterrows():
        print(f"\n{row['model']}:")
        for name, val in row["coefficients"].items():
            print(f"  {name}: {val:.6f}")

    os.makedirs("data", exist_ok=True)
    table.to_csv(REPORT_PATH, index=False)
    print(f"\nFull comparison saved to {REPORT_PATH}")
    print("No production files were modified: models/*.pkl, data/model_metrics.csv, "
          "app.py, and utils/* are unchanged.")


if __name__ == "__main__":
    main()
