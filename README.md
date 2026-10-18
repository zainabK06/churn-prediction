# Customer Churn Prediction & Retention Analytics Dashboard

**An end-to-end machine learning project that predicts telecom customer churn, explains model decisions, and turns risk scores into actionable retention insights through an interactive Streamlit dashboard.**

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![SHAP](https://img.shields.io/badge/Explainability-SHAP-2B6CB0)](https://shap.readthedocs.io/)

**Live demo:** https://churn-prediction-7l8f6ncxddwvlrethfpmir.streamlit.app/

---

## About the project

**Customer churn** happens when a customer stops using a company's service. For a telecom provider, this can mean a subscriber cancelling their phone or internet plan.

Not all customers have the same likelihood of leaving. Some are new, some have long-term contracts, and others use different services or payment methods. The **Customer Churn Prediction & Retention Dashboard** uses customer information to highlight potential churn risk and present it in a way that is useful for business decisions.

The application is designed to answer four questions:

- **Who might leave?** Identify customers with higher predicted churn scores.
- **What influences the predictions?** Explore customer characteristics that matter to the model.
- **How much monthly billing is associated with at-risk customers?** Compare monthly charges across risk groups.
- **Who should receive attention first?** View suggested outreach priorities and possible retention actions.

The dashboard is an educational demonstration using historical telecom customer data, not a live system connected to a real telecom company's customers.

## Explore the dashboard

The Streamlit application has four pages. No machine learning knowledge is needed to use the interface.

| Page | What it shows | Why it is useful |
|---|---|---|
| **Overview** | Customer totals, historical churn trends, factors influencing predictions, and a chart of customer risk groups | Understand the customer base and major churn patterns |
| **At-Risk Customers** | Customers ranked by predicted churn score, with filters, priority levels, and suggested actions | Find customers who may need attention |
| **Business Insights** | Monthly charges associated with different risk groups and customers grouped by outreach priority | Understand the billing exposure and plan follow-ups |
| **Customer Drill-Down** | A form that predicts churn risk for one customer based on their details | Explore an individual customer's risk assessment |

### 1. Overview: understand churn patterns

The Overview page combines two kinds of information:

- **Historical churn rates:** The share of customers who actually left, shown by contract type and length of time with the company (*tenure*).
- **Model insights:** The customer characteristics that most strongly influence predictions, plus the number of customers placed in each predicted risk group.

For example, the historical data shows that about **42.7% of month-to-month customers** churned, compared with about **2.8% of two-year-contract customers**. This is an association in the dataset, not proof that changing a contract will prevent churn.

The **Factors Influencing Churn Predictions** chart uses SHAP, an explanation technique. Longer bars indicate greater influence on the model's predictions overall. A longer bar does **not** automatically mean that a characteristic increases churn risk; its influence can go in either direction.

<img width="1920" height="2057" alt="image" src="https://github.com/user-attachments/assets/03108033-e4f6-4570-9363-0fff88eaf8c0" />

### 2. At-Risk Customers: see who may need attention

Each customer receives a churn-risk score from the model. The application groups scores into these simple dashboard tiers:

| Risk tier | Model score |
|---|---|
| **High** | 70% or above |
| **Medium** | 40% to below 70% |
| **Low** | Below 40% |

Customers can be filtered by risk tier and contract type. The table also displays monthly charges, customer value, outreach priority, and a suggested action.

These thresholds are **illustrative categories**, not scientifically validated cutoffs. A score of 80% should not be interpreted as a confirmed 80% real-world chance of churn.

<img width="1920" height="1273" alt="AdobeExpressPhotos_3d553a91444c4bc389b4f12f53da5fc9_CopyEdited" src="https://github.com/user-attachments/assets/87ae50a4-bf53-4d37-8507-9b7bba64a0f3" />

### 3. Business Insights: put risk into context

The Business Insights page adds a practical business perspective:

- **High-Risk Monthly Charges:** The combined monthly bills of customers in the High risk tier.
- **Revenue Exposure:** The percentage of all monthly charges in the analyzed group that comes from High-risk customers.
- **Monthly Charges by Risk Tier:** How much the company bills customers in the High, Medium, and Low risk groups.
- **Customers by Outreach Priority:** How many customers are marked Critical, High, or Routine for follow-up.

**Important:** High-risk monthly charges are **not revenue already lost** and are **not a prediction of how much money will be lost**. They show the amount of current monthly billing associated with customers flagged by the model. The source dataset does not establish a currency, so the app does not assume one.

#### How outreach priority works

Outreach priority considers both the model's risk tier and the customer's monthly charges. Customers with charges at or above the group's median are labeled **High Value**; others are **Standard Value**.

| Priority | Meaning |
|---|---|
| **Critical** | High-risk customer with relatively high monthly charges |
| **High** | High-risk customer with standard charges, or medium-risk customer with high charges |
| **Routine** | Other customers who can receive normal engagement and support |

The app suggests actions such as personalized outreach, onboarding assistance, satisfaction checks, or an optional contract incentive. These are **simple business rules**, not recommendations proven to stop churn.

<img width="1896" height="1793" alt="AdobeExpressPhotos_fd4d068a07f34cf18df8278e97517991_CopyEdited" src="https://github.com/user-attachments/assets/50d369f4-b4e2-4d6e-9fd1-066917c76c86" />

### 4. Customer Drill-Down: check an individual customer

A user can enter details such as contract type, tenure, services, payment method, and monthly charges. The app then displays:

- A **model churn score** and High/Medium/Low risk tier.
- A simple visual indicator of the score.
- A suggested outreach priority and retention action.

The form asks for familiar customer details. Additional calculated fields are generated automatically before the information is passed to the trained model.

<img width="1920" height="2017" alt="AdobeExpressPhotos_d7fda206333a4399994b959b0c629f31_CopyEdited" src="https://github.com/user-attachments/assets/acea0812-1096-4229-aa62-8e64fb887328" />

## Model performance at a glance

The dashboard uses a **Logistic Regression** model, a machine learning method that looks for patterns in past customer data and assigns scores to new customer records.

| Measure | Result | What it means |
|---|---:|---|
| Customers in the dataset | **7,043** | Historical customer records available for analysis |
| Historical churn rate | **26.5%** | Roughly one in four customers in the dataset left |
| Churn recall | **80.2%** | The model identified about eight out of ten actual churners at its classification threshold |
| Churn precision | **50.9%** | About half of customers predicted to churn actually did in the test data |
| ROC-AUC | **0.844** | Indicates good ability to rank customers who churned above those who stayed |
| Churners found in the top 20% of risk scores | **49.5%** | The highest-risk fifth of the test group contained almost half of its actual churners |

**What does the final result mean in practice?** In a held-out group of **1,409 customers**, the top **282 customers** ranked by model score included **185 of the 374 customers who actually churned**. This suggests that a focused outreach list can reach more potential churners than choosing customers at random.

Identifying a customer who later churns does not mean the company can necessarily prevent them from leaving. The model also produces false alarms, so scores should support human decisions rather than replace them.

## What's inside this repository?

The repository contains the dashboard, the customer data needed for the demonstration, the trained model, and Jupyter notebooks that document the supporting analysis.

```text
churn-prediction/
├── app/
│   └── streamlit_app.py
├── data/
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

*Notebook filenames may differ slightly in the repository (for example, some versions have `(1)` in their names). The app, dataset, and model paths must match the actual files.*

| File or folder | What it is for |
|---|---|
| `app/streamlit_app.py` | The web application, including charts, customer filters, predictions, and business insights |
| `data/processed/churn_engineered.csv` | The prepared customer dataset used by the application |
| `models/churn_model_phase2.pkl` | The saved, trained model that produces churn scores |
| `01_eda.ipynb` | Exploration of customer characteristics and churn patterns |
| `02_cleaning.ipynb` | Preparing and cleaning customer records |
| `03_model.ipynb` | Initial churn prediction model |
| `04_feature_engineering.ipynb` | Additional customer indicators derived from existing information |
| `05_imbalance_and_models.ipynb` | Model comparisons and handling the uneven number of churners and non-churners |
| `06_model_evaluation.ipynb` | Model performance and customer-ranking evaluation |
| `07_explainability.ipynb` | Explanation of the features influencing model predictions |
| `requirements.txt` | Python libraries needed to run the dashboard |

The notebooks are available for readers who want technical detail, but **they are not required to use the dashboard**.

## Run the dashboard locally

**Requirements:** Python and Git, with the model and processed dataset included in the repository.

```bash
# Get the repository
git clone https://github.com/zainabK06/churn-prediction.git
cd churn-prediction

# Create a Python environment
python -m venv .venv

# Install dependencies
python -m pip install -r requirements.txt

# Start the dashboard
streamlit run app/streamlit_app.py
```

Activate `.venv` before installing dependencies if needed: `.venv\Scripts\activate` on Windows, or `source .venv/bin/activate` on macOS/Linux. Open the local address printed in the terminal (usually `http://localhost:8501`).

## Technologies

**Python** powers the project. **pandas** and **NumPy** manage customer data; **scikit-learn** runs the trained Logistic Regression model; **SHAP** explains influential model features; **Streamlit** and power the interactive dashboard; and **joblib** loads the saved model. **Jupyter notebooks** provide additional technical detail for anyone interested in the analysis.

## Important limitations

- **Risk scores are estimates, not certainties.** The model uses balanced class weights, and its scores have not been calibrated as real-world churn probabilities.
- **Some high-risk predictions are incorrect.** The model is designed to find many potential churners, which also creates false positives.
- **Revenue exposure is not revenue loss.** Monthly charges are used as a simple indicator of customer billing value, not profit or guaranteed financial loss.
- **Suggested retention actions are illustrative.** The dataset does not show whether the suggested actions actually improve customer retention.
- **Not every chart uses the same group.** Historical customer trends use all **7,043** records; risk rankings and business insights use the **1,409-customer test group**.
- **The application is a portfolio demonstration.** It is not connected to live telecom systems or designed to make automated, high-stakes decisions.

---

**In short:** This project turns customer churn predictions into an understandable, interactive dashboard that connects customer behavior, model-based risk, monthly billing, and possible retention actions.
