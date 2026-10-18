# Customer Churn Prediction & Retention Analytics Dashboard

**An end-to-end machine learning project that predicts telecom customer churn, explains model decisions, and turns risk scores into actionable retention insights through an interactive Streamlit dashboard.**

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![SHAP](https://img.shields.io/badge/Explainability-SHAP-2B6CB0)](https://shap.readthedocs.io/)

**Repository:** [github.com/zainabK06/churn-prediction](https://github.com/zainabK06/churn-prediction)  
**Live demo:** https://churn-prediction-7l8f6ncxddwvlrethfpmir.streamlit.app/

---

## Project highlights

| Metric | Result |
|---|---:|
| Customer records | **7,043** |
| Observed churn rate | **26.5%** |
| Final model | **Class-weighted Logistic Regression** |
| Test ROC-AUC | **0.8439** |
| Churn-class recall | **80.21%** |
| Churn-class precision | **50.85%** |
| Churn-class F1 | **0.6224** |
| Average precision (PR analysis) | **0.6458** |
| Actual churners reached in top 20% of test customers | **49.47%** |
| Lift at 20% targeting | **2.47×** vs. random targeting |

> **Key takeaway:** Ranking the 1,409 held-out test customers by model score placed **185 of the 374 actual churners** within the highest-risk 282 customers (the top ~20%). This indicates useful targeting performance; identifying a customer does **not** mean preventing their churn.

## The problem

Customer churn occurs when subscribers stop using a company's service. For a telecom provider, understanding which customers are at risk can help focus customer-support and retention efforts where they may be most useful.

This project addresses three practical questions:

1. **Who is at risk?** Rank customers by the model's churn score.
2. **Why are they flagged?** Examine influential features with SHAP.
3. **What should the business do next?** Group customers by risk and billing value, estimate monthly charges associated with high-risk groups, and provide rule-based outreach suggestions.

## Interactive dashboard

The Streamlit application contains four pages:

| Page | Features |
|---|---|
| **Overview** | Historical churn statistics; SHAP feature-importance chart; churn rate by contract and tenure; predicted-risk distribution |
| **At-Risk Customers** | Ranked held-out customers, contract/risk filters, customer value, priority labels, and suggested outreach actions |
| **Business Insights** | High-risk monthly charges, share of monthly billing associated with high-risk customers, billing totals by risk tier, and outreach-priority counts |
| **Customer Drill-Down** | Individual customer input form, automatic feature engineering, churn score, risk classification, and suggested retention action |

**Screenshots:** Add actual app screenshots under `assets/` and insert them here once available. Suggested images are an Overview screenshot and a Business Insights screenshot. No screenshots are embedded by default to avoid broken links.

### How to interpret the business metrics

- **High-risk customers** are those with model scores of **70% or higher**; Medium is **40% to below 70%**, and Low is **below 40%**. These are *illustrative dashboard thresholds*, not calibrated real-world probabilities.
- **High-risk monthly charges** is the **sum of monthly bills** for customers in the High tier. It is **not an estimate of money already lost or guaranteed future losses**.
- **Outreach priority** combines model risk with billing value (High Value means monthly charges at or above the median in the analyzed test set). These priorities and recommended actions are **transparent rules**, not outputs of an independently trained recommendation model.
- The dataset does not establish a verified currency, so the dashboard does not assume a currency symbol.

## Dataset

The analysis uses a **telecom customer churn dataset** with **7,043 customers** and **21 original columns**, including customer demographics, tenure, subscribed services, contract, billing information, and churn outcome.

| Target class | Customers | Share |
|---|---:|---:|
| Did not churn | 5,174 | 73.5% |
| Churned | 1,869 | 26.5% |

The target is imbalanced: roughly one in four customers churned. The project uses a **stratified 80/20 train-test split** (`random_state=42`), yielding **5,634 training** and **1,409 test** customers. The 1,409-customer test set includes **374 churners**.

> The Overview page describes the full historical dataset; the risk-ranking, prioritization, and revenue-exposure demonstrations use the held-out test subset. These populations should not be confused.

## Methodology

### 1. Exploratory data analysis

Investigated the target distribution and relationships between churn and tenure, contract type, monthly charges, services, and billing options.

Notable descriptive findings:

- **Month-to-month customers:** ~**42.7%** churn, compared with **11.3%** on one-year contracts and **2.8%** on two-year contracts.
- **Tenure:** Customers who churned tended to have shorter tenure; the median tenure was approximately **10 months** for churners versus **38 months** for non-churners.
- **Payment:** Customers paying by electronic check had an observed churn rate of approximately **45.3%**.
- These are **associations**, not demonstrations of causality.

### 2. Cleaning and preprocessing

- Removed `customerID` from model inputs.
- Converted `TotalCharges` into numeric form and addressed missing/blank values.
- Converted binary categories into numerical values and used one-hot encoding for multi-category variables.
- Standardized inputs within the final Logistic Regression training pipeline using `StandardScaler`.
- Preserved a stratified held-out test set for final evaluation.

### 3. Feature engineering

Created features intended to represent customer lifecycle, services, and billing context:

| Feature | Description |
|---|---|
| `tenure_bucket` | Tenure bands: 0–6 months, 7–12 months, 13–24 months, 25+ months |
| `total_services` | Count of six internet add-on subscriptions |
| `has_internet` | Indicator for an active internet service |
| `high_risk_combo` | Indicator for **month-to-month contract + electronic-check payment** |
| `charge_vs_service_avg` | Difference between the customer's monthly bill and the mean bill for their internet-service group |

An exploratory feature comparison showed **only a marginal improvement in ROC-AUC** from engineered features. Their SHAP importance does not by itself establish that they improved predictive performance.

### 4. Class imbalance handling

Compared **balanced class weights**, **SMOTE oversampling**, and **random undersampling**. Their results were broadly similar. The final Logistic Regression uses `class_weight='balanced'`, prioritizing recall of customers who churn over minimizing false alerts.

### 5. Model comparison and selection

Compared Logistic Regression, Random Forest, and XGBoost using **5-fold stratified cross-validation**, then performed a focused `GridSearchCV` search on Logistic Regression.

**Held-out test-set results (positive class = churn):**

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| **Logistic Regression (selected)** | **0.5085** | **0.8021** | **0.6224** | 0.8439 |
| Random Forest | 0.5497 | 0.6364 | 0.5898 | 0.8204 |
| XGBoost | 0.5435 | 0.6845 | 0.6059 | 0.8228 |
| Tuned Logistic Regression | 0.5059 | 0.7995 | 0.6197 | **0.8451** |

**Why the original Logistic Regression?** Tuning delivered a very small increase in ROC-AUC, while the original model retained slightly better churn recall and F1. It was selected for a balance of churn detection, interpretability, and simplicity.

### 6. Ranking quality and business targeting

The final model achieved:

- **ROC-AUC:** 0.8439.
- **Average precision:** 0.6458, compared with the churn-prevalence baseline of approximately 0.2654.
- **Cumulative gains:** Targeting the top ~20% of customers by model score captured **185 / 374 = 49.47%** of actual churners in the held-out test set.
- **Cumulative lift:** **2.47×** at that targeting level compared with random selection in expectation.

This is useful for *ranking* a limited outreach list. It does not measure the impact or success rate of any retention campaign.

### 7. Explainability with SHAP

Used **SHAP's LinearExplainer** with a sample of standardized training observations as background to explain the final Logistic Regression model.

The global feature-importance and beeswarm plots highlighted features such as:

- Monthly charges relative to customers with the same internet-service type.
- Long-term contract indicators.
- Customer tenure and tenure bands.
- Streaming services, multiple lines, and fiber-optic internet.
- Electronic-check payment.

The local waterfall plot broke down a selected high-risk customer's model score into feature-level contributions, illustrating how SHAP can explain *why a customer was flagged*.

**Interpretation caution:** These contributions explain the trained model's behavior, **not causal drivers** of churn. For this linear explainer, contributions are in **log-odds**, not direct percentage-point changes in churn probability.

### 8. Business-facing application

The Streamlit dashboard presents model scores, risk tiers, billing exposure, customer prioritization, and simple retention suggestions in a format intended for nontechnical users. Business recommendations are **heuristics** based on available features; no campaign outcomes or retention effectiveness were measured.

## Tech stack

| Area | Tools |
|---|---|
| Language and data | Python, pandas, NumPy |
| Visualization and EDA | Matplotlib, Seaborn |
| Machine learning | scikit-learn, XGBoost, imbalanced-learn |
| Explainability | SHAP |
| Dashboard | Streamlit, Altair |
| Model persistence | joblib |
| Notebooks | Jupyter Notebook |
| Version control | Git and GitHub |

## Project structure

```text
churn-prediction/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── raw/
│   │   └── churn_data.csv
│   └── processed/
│       └── churn_engineered.csv
├── models/
│   └── churn_model_phase2.pkl
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_cleaning.ipynb
│   ├── 03_model.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_imbalance_and_models.ipynb
│   ├── 06_model_evaluation.ipynb
│   └── 07_explainability.ipynb
├── requirements.txt
└── README.md
```

*Structure shows the expected deployment layout. The filenames in your local/GitHub notebook directory may differ slightly; adjust this tree to match the repository if needed. The app specifically expects `models/churn_model_phase2.pkl` and `data/processed/churn_engineered.csv`.*

## Run locally

**Prerequisites:** Python and pip, plus the project files shown above.

```bash
# 1. Clone the repository
git clone https://github.com/zainabK06/churn-prediction.git
cd churn-prediction

# 2. Create a virtual environment
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
# source .venv/bin/activate

# 3. Install dependencies
python -m pip install -r requirements.txt

# 4. Launch the dashboard
streamlit run app/streamlit_app.py
```

Open the local address shown in your terminal (typically `http://localhost:8501`).

**Required runtime files:**

- `data/processed/churn_engineered.csv`
- `models/churn_model_phase2.pkl`

If `data/` and `models/` are ignored by Git, these two files must still be present in the deployed repository or provisioned through another supported source. The app will not run without them.

**Model artifact note:** The attached training and explainability notebooks save/load `churn_model_final.pkl`, whereas the Streamlit app expects `churn_model_phase2.pkl`. The file deployed under the latter name must be the **same selected fitted Logistic Regression pipeline**, not a different model or a newly trained approximation.

## Limitations and responsible interpretation

- **One historical dataset:** Performance may not generalize to other telecom companies, periods, or customer populations.
- **False positives:** The model is recall-oriented; some customers flagged as high-risk will not actually churn. For example, the churn-class test precision is approximately 50.9%.
- **Uncalibrated scores:** Balanced class weights can distort probability calibration. Scores and the 40%/70% dashboard risk tiers should be viewed as **prioritization indicators**, not verified real-world probabilities.
- **No causal claims:** SHAP and descriptive correlations do not demonstrate that changing a contract, service, or payment method will prevent churn.
- **No measured intervention benefit:** Revenue exposure is monthly billing associated with high-risk customers, **not expected or realized losses**. Outreach actions are suggested rules and have not been evaluated experimentally.
- **Demo data:** The at-risk and business insights pages use the **1,409 held-out test customers**; row identifiers in the app are demo identifiers rather than original customer IDs.
- **Reproducibility:** Feature transformations used for new individual-customer predictions should remain consistent with the transformations applied during training.

## What I learned

This beginner end-to-end project strengthened my understanding of how to:

- Explore and clean tabular data, and reason about an imbalanced classification target.
- Design and assess engineered features instead of assuming every new feature improves performance.
- Compare models with stratified cross-validation and evaluate precision–recall trade-offs.
- Evaluate a model not just by accuracy or ROC-AUC but also by **how well it ranks customers for limited outreach capacity**.
- Explain global and individual model predictions with SHAP.
- Turn notebook results into a working Streamlit application with transparent business rules.

---

**Project scope:** An educational, beginner-level end-to-end machine learning and analytics application. It is designed to demonstrate practical workflow and analytical reasoning, not to serve as a production-validated retention decision system.
