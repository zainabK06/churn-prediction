import streamlit as st
import pandas as pd
import joblib

# loading saved models
model = joblib.load("../models/churn_model.pkl")
scaler = joblib.load("../models/scaler.pkl")
feature_columns = joblib.load("../models/feature_columns.pkl")

st.title("Customer Churn Prediction")

st.write("Enter customer details to predict whether the customer is likely to churn.")

# Columns that were converted from Yes/No to 1/0
binary_cols = [
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling"
]

# Columns that were one-hot encoded during training
one_hot_cols = [
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaymentMethod",
    "gender"
]

# Numerical columns that were scaled during training
numeric_features = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]




# ============================================================
# 6. CUSTOMER INFORMATION
# ============================================================

st.header("Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

with col2:
    senior_citizen = st.selectbox(
        "Senior Citizen",
        [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

with col3:
    partner = st.selectbox(
        "Partner",
        ["No", "Yes"]
    )


col1, col2, col3 = st.columns(3)

with col1:
    dependents = st.selectbox(
        "Dependents",
        ["No", "Yes"]
    )

with col2:
    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=72,
        value=12,
        step=1
    )

with col3:
    phone_service = st.selectbox(
        "Phone Service",
        ["No", "Yes"]
    )


# ============================================================
# 7. SERVICES
# ============================================================

st.header("Services")

col1, col2, col3 = st.columns(3)

with col1:
    multiple_lines = st.selectbox(
        "Multiple Lines",
        [
            "No",
            "No phone service",
            "Yes"
        ]
    )

with col2:
    internet_service = st.selectbox(
        "Internet Service",
        [
            "DSL",
            "Fiber optic",
            "No"
        ]
    )

with col3:
    online_security = st.selectbox(
        "Online Security",
        [
            "No",
            "No internet service",
            "Yes"
        ]
    )


col1, col2, col3 = st.columns(3)

with col1:
    online_backup = st.selectbox(
        "Online Backup",
        [
            "No",
            "No internet service",
            "Yes"
        ]
    )

with col2:
    device_protection = st.selectbox(
        "Device Protection",
        [
            "No",
            "No internet service",
            "Yes"
        ]
    )

with col3:
    tech_support = st.selectbox(
        "Tech Support",
        [
            "No",
            "No internet service",
            "Yes"
        ]
    )


col1, col2, col3 = st.columns(3)

with col1:
    streaming_tv = st.selectbox(
        "Streaming TV",
        [
            "No",
            "No internet service",
            "Yes"
        ]
    )

with col2:
    streaming_movies = st.selectbox(
        "Streaming Movies",
        [
            "No",
            "No internet service",
            "Yes"
        ]
    )

with col3:
    contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )


# ============================================================
# 8. BILLING INFORMATION
# ============================================================

st.header("Billing Information")

col1, col2, col3 = st.columns(3)

with col1:
    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["No", "Yes"]
    )

with col2:
    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

with col3:
    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0,
        step=1.0
    )


total_charges = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=840.0,
    step=1.0
)


st.divider()


# ============================================================
# 9. PREDICTION BUTTON
# ============================================================

predict_button = st.button(
    "🔍 Predict Churn",
    use_container_width=True
)


if predict_button:

    input_data = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [senior_citizen],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "PhoneService": [phone_service],
        "MultipleLines": [multiple_lines],
        "InternetService": [internet_service],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protection],
        "TechSupport": [tech_support],
        "StreamingTV": [streaming_tv],
        "StreamingMovies": [streaming_movies],
        "Contract": [contract],
        "PaperlessBilling": [paperless_billing],
        "PaymentMethod": [payment_method],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges]
    })

    # binary encoding
    for col in binary_cols:
        input_data[col] = input_data[col].map({
            "Yes": 1,
            "No": 0
        })

    # one-hot encoding
    input_data = pd.get_dummies(
        input_data,
        columns=one_hot_cols,
        drop_first=False
    )   

    input_data = input_data.reindex(
        columns=feature_columns,
        fill_value=0
    )

    input_data = input_data.astype(float)

    # Scaling numerical features
    input_data[numeric_features] = scaler.transform(
        input_data[numeric_features]
    )

    # --------------------------------------------------------
    # Make prediction
    # --------------------------------------------------------

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    st.divider()

    st.header("Prediction Result")


    if prediction == 1:

        st.error(
            "⚠️ The customer is predicted to churn."
        )

    else:

        st.success(
            "✅ The customer is predicted to stay."
        )

    # --------------------------------------------------------
    # Probability
    # --------------------------------------------------------

    st.metric(
        label="Churn Probability",
        value=f"{probability:.2%}"
    )


    # --------------------------------------------------------
    # Progress bar
    # --------------------------------------------------------

    st.progress(
        float(probability)
    )


    # --------------------------------------------------------
    # Simple interpretation
    # --------------------------------------------------------

    if probability >= 0.5:

        st.write(
            "The model predicts a higher probability of churn "
            "for this customer."
        )

    else:

        st.write(
            "The model predicts a higher probability of the "
            "customer staying."
        )