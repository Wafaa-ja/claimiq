# ClaimIQ — Motor Insurance Claim Frequency and Pure Premium Simulator

An actuarial decision-support tool comparing classical statistical models and machine learning techniques for motor insurance claim frequency prediction.

## 🌐 Live Demo

[![Launch ClaimIQ](https://img.shields.io/badge/Launch-ClaimIQ-0A84FF?style=for-the-badge)](https://claimiq-app.streamlit.app)

## 📸 Application Preview

<p align="center">
  <img src="images/home-v2.png" width="900" alt="ClaimIQ application homepage">
</p>

---

## Project Overview

ClaimIQ estimates expected motor insurance claim frequency from four fitted models:

| Model | Type | Test MAE | AIC |
|-------|------|----------|-----|
| Poisson GLM | Statistical | 0.09888 | 229,548.83 |
| Negative Binomial GLM | Statistical | 0.09945 | 228,837.56 |
| Random Forest | Machine Learning | 0.09809 | N/A |
| XGBoost | Machine Learning | 0.09821 | N/A |

Models are trained on the French Motor Third-Party Liability (MTPL) dataset
comprising 678,013 policy records.

---

## Project Structure

```
claimiq_app/
├── app.py                      # Main Streamlit application
├── save_models.py              # Train and save all four models
├── requirements.txt
├── README.md
├── models/
│   ├── poisson_model.pkl
│   ├── negative_binomial_model.pkl
│   ├── random_forest_model.pkl
│   ├── xgboost_model.pkl
│   └── feature_metadata.json
├── data/
│   └── model_metrics.csv
├── utils/
│   ├── __init__.py
│   ├── model_loader.py
│   ├── prediction.py
│   ├── validation.py
│   └── formatting.py
└── assets/
    └── logo.png  (optional)
```

---

## Installation

```bash
# 1. Clone or download this project
cd claimiq_app

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Step 1 — Save the Models

Before launching the app, you must train and save the four models.

Place your `freMTPL2freq.csv` dataset anywhere accessible, then run:

```bash
python save_models.py --data path/to/freMTPL2freq.csv
```

This will:
- Train all four models using the same train/test split (random_state=42)
- Save fitted models to the `models/` folder
- Save evaluation metrics to `data/model_metrics.csv`
- Save feature metadata to `models/feature_metadata.json`

Expected output:
```
Loading dataset from: ...
Train: 542,410  |  Test: 135,603
Fitting Poisson GLM...
  MAE=0.09888  AIC=229548.83
  Saved to models/poisson_model.pkl
Fitting Negative Binomial GLM (alpha=1)...
  MAE=0.09945  AIC=228837.56
  Saved to models/negative_binomial_model.pkl
Fitting Random Forest...
  MAE=0.09809
  Saved to models/random_forest_model.pkl
Fitting XGBoost...
  MAE=0.09821
  Saved to models/xgboost_model.pkl
Metrics saved to data/model_metrics.csv
Feature metadata saved to models/feature_metadata.json
=== All models saved successfully ===
```

---

## Step 2 — Launch the Application

```bash
streamlit run app.py
```

The application will open at `http://localhost:8501`.

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

## Actuarial Formulas

### Claim Frequency (GLMs)
```
log(μᵢ) = β₀ + β₁·BonusMalus + β₂·DrivAge + β₃·VehAge + β₄·Density + log(Exposure)
```

Poisson GLM coefficients from the French MTPL dataset:
- Intercept: −3.7115
- BonusMalus: +0.02289
- DrivAge: +0.00687
- VehAge: −0.04281
- Density: +9.68 × 10⁻⁶

### Annualised Frequency
```
Annualised Frequency = Predicted Claims / Exposure
```

### Pure Premium
```
Pure Premium = Predicted Claim Frequency × Average Claim Cost
```

### Loaded Premium (optional)
```
Loaded Premium = Pure Premium / (1 − Expense Ratio − Profit Margin)
```

---

## Input Ranges (Training Data)

| Variable | Min | Max | Dataset Mean |
|----------|-----|-----|--------------|
| DrivAge | 18 | 100 | 45.5 |
| VehAge | 0 | 100 | 7.0 |
| BonusMalus | 50 | 230 | 59.8 |
| Density | 1 | 27,000 | 1,792 |
| Exposure | 0.01 | 2.00 | 0.53 |

---

## Deployment to Streamlit Community Cloud

1. Push the project to a GitHub repository.
2. Ensure `requirements.txt` is in the root.
3. Add model `.pkl` files to the `models/` folder and commit them,
   or use `st.secrets` + cloud storage for large files.
4. Connect the repository at https://share.streamlit.io.
5. Set the main file path to `app.py`.

---

## Limitations

- Trained on one national dataset (French MTPL); findings may not generalise.
- Only claim frequency is modelled. Full pricing requires claim severity.
- The tool is a proof-of-concept and must not be used for actual quotations.

---

## Disclaimer

This application is provided for research and educational purposes only.
All outputs are illustrative. They must not be treated as insurance quotations,
underwriting recommendations, or official actuarial estimates.

---

## References

- Noll, A., Salzmann, R., & Wüthrich, M. V. (2018). *Case study: French MTPL claims.* SSRN.
- Denuit, M., et al. (2007). *Actuarial Modelling of Claim Counts.* Wiley.
- Goldburd, M., et al. (2025). *GLMs for Insurance Rating.* CAS.
- Henckaerts, R., et al. (2021). North American Actuarial Journal, 25(2).
- Breiman, L. (2001). *Machine Learning*, 45(1).
- Chen, T., & Guestrin, C. (2016). KDD 2016.
