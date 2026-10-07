<h1 align="center">ClaimIQ</h1>

<p align="center">
<b>Actuarial Pricing Analysis of Motor Insurance Claim Frequency</b>
</p>

<p align="center">
A comparative actuarial modeling study evaluating Generalized Linear Models and machine learning approaches for motor insurance claim frequency prediction.
</p>

<p align="center">
<a href="https://claimiq-app.streamlit.app">
<img src="https://img.shields.io/badge/🚀_Live_Demo-ClaimIQ-0A84FF?style=for-the-badge">
</a>

<img src="https://img.shields.io/badge/Python-3.13-blue?style=for-the-badge&logo=python">

<img src="https://img.shields.io/badge/Streamlit-App-red?style=for-the-badge&logo=streamlit">

<img src="https://img.shields.io/badge/Machine_Learning-XGBoost-success?style=for-the-badge">
</p>

<p align="center">
  <img src="images/home-v3.png" width="950">
</p>

##  Live Demo

[![Launch ClaimIQ](https://img.shields.io/badge/Launch-ClaimIQ-0A84FF?style=for-the-badge)](https://claimiq-app.streamlit.app)
## Analysis Capabilities

- Estimate motor insurance claim frequency using four competing models.
- Compare Poisson and Negative Binomial GLMs with Random Forest and XGBoost.
- Evaluate out-of-sample predictive performance using MAE.
- Compare statistical model fit using AIC.
- Validate model rankings across 20 independent 80/20 train-test splits.
- Examine GLM coefficients and machine-learning feature importance.
- Generate policy-level frequency predictions.
- Translate predicted frequency into illustrative pure premiums.
- Perform scenario analysis across policyholder and vehicle characteristics.
- Explore the trade-off between predictive performance and actuarial interpretability.

## Project Overview

ClaimIQ is an actuarial pricing analytics project that compares traditional Generalized Linear Models with machine learning approaches for predicting motor insurance claim frequency.

Using 678,013 policy records from the French Motor Third-Party Liability (MTPL) dataset, the study compares:

- Poisson GLM
- Negative Binomial GLM
- Random Forest
- XGBoost

Rather than relying on a single train-test comparison, the analysis evaluates model performance across 20 independent 80/20 splits to investigate whether any predictive advantage offered by machine learning is meaningful and reproducible.

The project also considers an important actuarial modeling trade-off: **predictive accuracy versus interpretability, statistical inference, and transparency**. This allows model selection to be considered not only from a predictive perspective, but also in the context of practical actuarial pricing.
---

## Project Structure

```
app.py                        # Thin entrypoint + page router (~35 lines)
claimiq/                      # App package: theme, components, charts, icons, data loader, pages/
  theme.py                    # Design tokens (light/dark) + CSS injection
  components.py                # card, kpi, data_table, badge, etc.
  charts.py                    # Shared Plotly template
  data.py                      # Loads data/research_results.json (single source of truth for all displayed metrics)
  pages/                       # One module per page (home, prediction, pure_premium, scenario, model_comparison, about)
save_models.py                # Model training (incl. NB2 dispersion-parameter estimation)
nb2_validation.py             # NB2 profile-likelihood MLE validation
validation_robustness.py      # 20-split repeated-validation script
generate_research_results.py  # Builds data/research_results.json from the CSVs/models below
models/                       # Saved ML models
data/                         # Dataset, metrics, and validation result CSVs/JSON
images/                       # README screenshots
utils/                        # Prediction/validation/formatting helpers (used by claimiq/)
```

## Installation

```bash
git clone https://github.com/Wafaa-ja/claimiq.git
cd claimiq
pip install -r requirements.txt
streamlit run app.py
```


---



---

## Application Pages

| Page | Description |
|------|-------------|
| Home | Overview and quick-start |
| Claim Frequency Prediction | Enter policyholder profile, get predictions from all four models |
| Pure Premium Calculator | Convert predicted frequency to an illustrative pure premium |
| Scenario Analysis | Vary one input and observe its effect on predicted frequency |
| Model Comparison | MAE table, AIC comparison, feature importance charts |
| About the Research | Dataset, methodology, findings, references, limitations |

---

---

## Technologies Used

| Category | Technologies |
|----------|--------------|
| Language | Python 3.13 |
| Web Framework | Streamlit |
| Machine Learning | Scikit-learn, XGBoost |
| Statistical Modeling | Statsmodels (Poisson GLM, Negative Binomial GLM) |
| Data Processing | Pandas, NumPy |
| Visualization | Plotly |

---

## Dataset

The analysis uses the French Motor Third-Party Liability (MTPL) dataset, a widely used benchmark dataset for actuarial claim-frequency modeling.

- **678,013 policy records**
- **80/20 train-test split**
- **94.98% zero-claim policies**
- Count-based response variable representing motor insurance claim frequency

The high proportion of zero-claim policies reflects the characteristics of insurance frequency data and motivates the use and comparison of count-based statistical models.

---

## Future Improvements

- Predict claim severity in addition to claim frequency.
- Implement SHAP values for model explainability.
- Support additional pricing models.
- Deploy with Docker and cloud infrastructure.
- Expand the simulator with more insurance datasets.

---

## Author

**Wafaa Jawad**  
B.S. Actuarial Science — King Fahd University of Petroleum and Minerals (KFUPM)

Actuarial science student interested in insurance analytics, statistical modeling, risk analysis, and the application of machine learning to actuarial problems.
