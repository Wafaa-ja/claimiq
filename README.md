<h1 align="center">
🚗 ClaimIQ
</h1>

<p align="center">
<b>Motor Insurance Claim Frequency & Pure Premium Simulator</b>
</p>

<p align="center">
An actuarial decision-support application comparing classical statistical models and machine learning techniques.
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
<img src="images/home-v2.png" width="100%">
</p>

## 🌐 Live Demo

[![Launch ClaimIQ](https://img.shields.io/badge/Launch-ClaimIQ-0A84FF?style=for-the-badge)](https://claimiq-app.streamlit.app)
## ✨ Features

- Predict motor insurance claim frequency using four fitted models.
- Compare Poisson GLM, Negative Binomial GLM, Random Forest, and XGBoost.
- Estimate illustrative pure premiums.
- Explore what-if scenarios using interactive sliders.
- Compare model performance using MAE and AIC.
- View feature importance and GLM coefficients.
- Built with Streamlit for an intuitive web interface.

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

## 📂 Project Structure

```
app.py                # Main Streamlit application
save_models.py        # Model training
models/               # Saved ML models
data/                 # Dataset & metrics
images/               # README screenshots
utils/                # Helper functions
```

## 🚀 Installation

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
