from pathlib import Path

import joblib
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Customer Churn Dashboard", page_icon="📊", layout="wide")

# Accent colors for charts only; Streamlit handles background and text colors.
NAVY = "#12243A"
TEAL = "#159A93"
CORAL = "#E77967"
BLUE = "#5486B6"
GOLD = "#D9A34B"
MUTED = "#68798B"
# Let Streamlit control its normal page background, sidebar, and text colors.
# The bordered chart containers below provide visual separation without theme overrides.
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "churn_engineered.csv"
MODEL_PATH = ROOT / "models" / "churn_model_final.pkl"


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)  # Only load a model file you trust.


try:
    df = load_data()
    model = load_model()
except (FileNotFoundError, OSError) as exc:
    st.error(f"Required data/model file could not be loaded: {exc}")
    st.stop()

if "Churn" not in df.columns:
    st.error("The engineered CSV must contain the 'Churn' column.")
    st.stop()

X = df.drop(columns="Churn")
y = df["Churn"]
if not hasattr(model, "feature_names_in_"):
    st.error("Saved model does not contain feature names. Save the fitted Phase 2 pipeline from Notebook 05.")
    st.stop()
FEATURES = list(model.feature_names_in_)
if list(X.columns) != FEATURES:
    st.error("Feature names/order in churn_engineered.csv do not match the saved Phase 2 model. Check Notebook 04 and 05 outputs.")
    st.stop()

# Same held-out partition as Notebooks 05 and 06, assuming CSV/order unchanged.
X_train, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


def contract_labels(frame):
    return np.select(
        [frame["Contract_One year"].eq(1), frame["Contract_Two year"].eq(1)],
        ["One year", "Two year"], default="Month-to-month"
    )


def internet_labels(frame):
    return np.select(
        [frame["InternetService_Fiber optic"].eq(1), frame["InternetService_No"].eq(1)],
        ["Fiber optic", "No"], default="DSL"
    )


def derive_reference_averages(engineered):
    # Reconstruct the fixed InternetService-group means used in Notebook 04.
    temp = engineered.copy()
    temp["Internet_Group"] = internet_labels(temp)
    temp["Recovered_Mean"] = temp["MonthlyCharges"] - temp["charge_vs_service_avg"]
    spread = temp.groupby("Internet_Group")["Recovered_Mean"].agg(lambda s: s.max() - s.min())
    if (spread > 1e-4).any():
        raise ValueError("Recovered InternetService-group means are not constant. Verify Notebook 04's feature formula.")
    return temp.groupby("Internet_Group")["Recovered_Mean"].mean().to_dict()


try:
    SERVICE_AVERAGES = derive_reference_averages(df)
except (KeyError, ValueError) as exc:
    st.error(f"Feature-engineering consistency check failed: {exc}")
    st.stop()


# Use exact known dummy names; missing category dummy = reference category.
CATEGORICAL = [
    "gender", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaymentMethod"
]
BINARY = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
ADDONS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]


def prepare_customer(raw):
    """Reproduce Notebook 04's numerical, binary, dummy and engineered columns."""
    values = dict.fromkeys(FEATURES, 0.0)
    for name in ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]:
        values[name] = float(raw[name])
    for name in BINARY:
        values[name] = float(raw[name] == "Yes")

    for name in CATEGORICAL:
        dummy = f"{name}_{raw[name]}"
        if dummy in values:
            values[dummy] = 1.0
        elif not any(c.startswith(f"{name}_") for c in FEATURES):
            raise ValueError(f"No encoded columns found for {name}; check Notebook 04.")
        # Otherwise the selected category may be the one dropped by drop_first=True.

    months = int(raw["tenure"])
    # Confirm the interval boundaries against Notebook 04 before relying on form outputs.
    if months <= 6:
        bucket = "0-6mo"
    elif months <= 12:
        bucket = "6-12mo"
    elif months <= 24:
        bucket = "1-2yr"
    else:
        bucket = "2+yr"
    dummy = f"tenure_bucket_{bucket}"
    if dummy in values:
        values[dummy] = 1.0

    values["total_services"] = float(sum(raw[name] == "Yes" for name in ADDONS))
    values["has_internet"] = float(raw["InternetService"] != "No")
    values["high_risk_combo"] = float(
        raw["Contract"] == "Month-to-month" and raw["PaymentMethod"] == "Electronic check"
    )
    values["charge_vs_service_avg"] = float(
        raw["MonthlyCharges"] - SERVICE_AVERAGES[raw["InternetService"]]
    )
    return pd.DataFrame([values], columns=FEATURES).astype(float)


def risk_level(score):
    # Display tiers only; these cutoffs were not tuned or calibrated.
    if score >= 0.70:
        return "High"
    if score >= 0.40:
        return "Medium"
    return "Low"


# Step 2.7: transparent business heuristics on held-out demonstration customers.
# MonthlyCharges is recurring bill amount, NOT observed profit or actual loss.
RISK_SCORES = model.predict_proba(X_test)[:, 1]
risk_df = X_test.copy()
risk_df["Churn_Probability"] = RISK_SCORES
risk_df["Actual_Churn"] = y_test
risk_df["Contract_Type"] = contract_labels(risk_df)
risk_df["Risk_Level"] = risk_df["Churn_Probability"].apply(risk_level)
risk_df["Demo_Row_ID"] = risk_df.index + 1
VALUE_THRESHOLD = float(risk_df["MonthlyCharges"].median())


def customer_value(monthly_charge):
    return "High Value" if monthly_charge >= VALUE_THRESHOLD else "Standard Value"


def customer_priority(risk, value):
    if risk == "High" and value == "High Value":
        return "Critical"
    if risk == "High" or (risk == "Medium" and value == "High Value"):
        return "High"
    return "Routine"


def retention_action(risk, priority, contract, tenure):
    # Suggested, rule-based outreach—not evidence that an action prevents churn.
    if priority == "Critical":
        return "Prioritize personalized retention outreach"
    if risk == "High" and contract == "Month-to-month":
        return "Consider a voluntary longer-term plan incentive"
    if risk == "High" and tenure <= 6:
        return "Offer onboarding support and a satisfaction check"
    if risk == "High":
        return "Ask about service concerns and offer assistance"
    if risk == "Medium":
        return "Check satisfaction and follow up if needed"
    return "Continue regular customer engagement"


risk_df["Customer_Value"] = risk_df["MonthlyCharges"].apply(customer_value)
risk_df["Priority"] = [
    customer_priority(risk, value)
    for risk, value in zip(risk_df["Risk_Level"], risk_df["Customer_Value"])
]
risk_df["Recommended_Action"] = [
    retention_action(risk, priority, contract, tenure)
    for risk, priority, contract, tenure in zip(
        risk_df["Risk_Level"], risk_df["Priority"], risk_df["Contract_Type"], risk_df["tenure"]
    )
]
risk_df = risk_df.sort_values("Churn_Probability", ascending=False)

HIGH_RISK_REVENUE = float(
    risk_df.loc[risk_df["Risk_Level"].eq("High"), "MonthlyCharges"].sum()
)
TOTAL_MONTHLY_REVENUE = float(risk_df["MonthlyCharges"].sum())
REVENUE_EXPOSURE = (
    100 * HIGH_RISK_REVENUE / TOTAL_MONTHLY_REVENUE
    if TOTAL_MONTHLY_REVENUE > 0 else 0.0
)

# Horizontal navigation bar instead of a sidebar.
st.markdown("# Churn Prediction Dashboard")
page = st.radio(
    "Dashboard pages",
    ["Overview", "At-Risk Customers", "Business Insights", "Customer Drill-Down"],
    horizontal=True,
    label_visibility="collapsed",
)
st.divider()

# Global model explanations for the same reference method used in Notebook 07:
# independent-feature linear SHAP = coefficient * (scaled x - background mean).
def global_feature_importance(training_features, test_features, coefficients, scaler):
    train_scaled = pd.DataFrame(
        scaler.transform(training_features), columns=training_features.columns,
        index=training_features.index,
    )
    test_scaled = pd.DataFrame(
        scaler.transform(test_features), columns=test_features.columns,
        index=test_features.index,
    )
    background = train_scaled.sample(n=min(200, len(train_scaled)), random_state=42)
    contributions = (test_scaled - background.mean(axis=0)) * coefficients
    imp = contributions.abs().mean(axis=0).sort_values(ascending=False).head(10)
    return imp.rename_axis("Feature").reset_index(name="Importance")


def named_feature(feature):
    names = {
        "charge_vs_service_avg": "Charges vs. service average",
        "Contract_Two year": "Two-year contract",
        "Contract_One year": "One-year contract",
        "tenure_bucket_2+yr": "Tenure over two years",
        "tenure_bucket_1-2yr": "Tenure one to two years",
        "tenure": "Customer tenure",
        "StreamingMovies_Yes": "Streaming movies",
        "StreamingTV_Yes": "Streaming TV",
        "MultipleLines_Yes": "Multiple phone lines",
        "InternetService_Fiber optic": "Fiber-optic internet",
        "PaymentMethod_Electronic check": "Electronic-check payment",
        "total_services": "Number of add-on services",
        "PaperlessBilling": "Paperless billing",
        "MonthlyCharges": "Monthly charges",
    }
    return names.get(feature, feature.replace("_", " "))


def rate_chart(values, label_color=TEAL):
    chart_frame = values.rename_axis("Group").reset_index(name="Churn Rate (%)")
    return (alt.Chart(chart_frame)
        .mark_bar(color=label_color, cornerRadiusEnd=6, size=32)
        .encode(
            x=alt.X("Group:N", sort=None, title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Churn Rate (%):Q", title="Historical churn rate (%)", scale=alt.Scale(domain=[0, 100])),
            tooltip=["Group:N", alt.Tooltip("Churn Rate (%):Q", format=".1f")],
        )
        .properties(height=235)
    )


if page == "Overview":
    st.title("Customer Churn Overview")
    st.caption("Understand historical churn patterns and what drives your model's predictions.")
    a, b, c = st.columns(3)
    a.metric("Total Customers", f"{len(df):,}")
    b.metric("Historical Churn Rate", f"{df['Churn'].mean():.1%}")
    c.metric("Customers Who Churned", f"{int(df['Churn'].sum()):,}")

    st.write("")
    with st.container(border=True):
        st.subheader("Most Influential Factors in Churn Predictions")
        st.caption("Based on the trained model's SHAP contributions across held-out customers. Larger bars indicate greater influence, not higher churn risk.")
        try:
            fitted_scaler = model.named_steps["standardscaler"]
            fitted_classifier = model.named_steps["logisticregression"]
            importance = global_feature_importance(
                X_train, X_test, fitted_classifier.coef_[0], fitted_scaler
            )
            importance["Label"] = importance["Feature"].map(named_feature)
            chart = (alt.Chart(importance)
                .mark_bar(color=TEAL, cornerRadiusEnd=5)
                .encode(
                    x=alt.X("Importance:Q", title="Average absolute SHAP impact (log-odds)"),
                    y=alt.Y("Label:N", sort="-x", title=None),
                    tooltip=[alt.Tooltip("Label:N", title="Feature"),
                             alt.Tooltip("Importance:Q", title="Mean |SHAP|", format=".3f")],
                ).properties(height=370)
            )
            st.altair_chart(chart, use_container_width=True, theme="streamlit")
            st.caption("SHAP explains the model's predictions, not the causes of churn. Related features can share overlapping information.")
        except (AttributeError, KeyError, ValueError) as exc:
            st.warning(f"Unable to compute model feature importance: {exc}")

    overview = df.copy()
    overview["Contract_Type"] = contract_labels(overview)
    contract_order = ["Month-to-month", "One year", "Two year"]
    contract_rates = overview.groupby("Contract_Type")["Churn"].mean().reindex(contract_order).mul(100)
    required_buckets = ["tenure_bucket_6-12mo", "tenure_bucket_1-2yr", "tenure_bucket_2+yr"]
    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            st.subheader("Churn by Contract")
            st.caption("Share of customers who actually churned, by contract type.")
            st.altair_chart(rate_chart(contract_rates, BLUE), use_container_width=True, theme="streamlit")
    with right:
        with st.container(border=True):
            st.subheader("Churn by Tenure")
            st.caption("Historical churn across the tenure groups from Notebook 04.")
            if all(col in overview.columns for col in required_buckets):
                overview["Tenure_Group"] = np.select(
                    [overview[col].eq(1) for col in required_buckets],
                    ["6–12 months", "1–2 years", "Over 2 years"],
                    default="0–6 months",
                )
                tenure_order = ["0–6 months", "6–12 months", "1–2 years", "Over 2 years"]
                tenure_rates = overview.groupby("Tenure_Group")["Churn"].mean().reindex(tenure_order).mul(100)
                st.altair_chart(rate_chart(tenure_rates, CORAL), use_container_width=True, theme="streamlit")
            else:
                st.warning("Tenure-bucket columns differ from Notebook 04's expected names.")

    with st.container(border=True):
        st.subheader("Model Risk Distribution")
        st.caption("Distribution of predicted risk categories among the 1,409 held-out customers (not the full historical dataset).")
        scores = model.predict_proba(X_test)[:, 1]
        levels = pd.Series(scores).apply(risk_level).value_counts()
        risk_counts = pd.DataFrame({
            "Risk Tier": ["High", "Medium", "Low"],
            "Customers": [int(levels.get(tier, 0)) for tier in ["High", "Medium", "Low"]],
        })
        bar = alt.Chart(risk_counts).mark_bar(cornerRadiusEnd=5, size=38).encode(
            x=alt.X("Risk Tier:N", sort=["High", "Medium", "Low"], title=None),
            y=alt.Y("Customers:Q", title="Number of customers"),
            color=alt.Color("Risk Tier:N", scale=alt.Scale(
                domain=["High", "Medium", "Low"], range=[CORAL, GOLD, TEAL]
            ), legend=None),
            tooltip=["Risk Tier:N", "Customers:Q"],
        ).properties(height=230)
        st.altair_chart(bar, use_container_width=True, theme="streamlit")
        st.caption("Illustrative risk tiers: High ≥ 70%, Medium 40–69.9%, Low < 40%. Scores are not calibrated probabilities.")

elif page == "At-Risk Customers":
    st.title("At-Risk Customers")
    st.caption("Ranked model scores for the 1,409 held-out test customers, not live customer accounts.")
    risk = risk_df

    a, b, c = st.columns(3)
    a.metric("High Risk", int(risk["Risk_Level"].eq("High").sum()))
    b.metric("Medium Risk", int(risk["Risk_Level"].eq("Medium").sum()))
    c.metric("Low Risk", int(risk["Risk_Level"].eq("Low").sum()))
    st.caption("High ≥ 70%, Medium 40–69.9%, Low < 40%. These labels are for display, not validated targeting thresholds.")

    with st.container(border=True):
        st.subheader("Find Customers")
        left, right = st.columns(2)
        with left:
            selected_risk = st.selectbox("Risk Level", ["All", "High", "Medium", "Low"])
        with right:
            selected_contract = st.selectbox("Contract", ["All", "Month-to-month", "One year", "Two year"])
        filtered = risk.copy()
        if selected_risk != "All":
            filtered = filtered.loc[filtered["Risk_Level"].eq(selected_risk)]
        if selected_contract != "All":
            filtered = filtered.loc[filtered["Contract_Type"].eq(selected_contract)]
        st.write(f"Showing **{len(filtered):,}** customers, sorted by predicted churn score.")
        view = [
            "Demo_Row_ID", "Contract_Type", "tenure", "MonthlyCharges",
            "Churn_Probability", "Risk_Level", "Customer_Value",
            "Priority", "Recommended_Action"
        ]
        st.dataframe(
            filtered[view], hide_index=True, use_container_width=True,
            column_config={
                "Demo_Row_ID": "Demo Row ID",
                "Contract_Type": "Contract",
                "tenure": "Tenure (months)",
                "MonthlyCharges": st.column_config.NumberColumn("Monthly Charges", format="%.2f"),
                "Churn_Probability": st.column_config.ProgressColumn("Model Churn Score", min_value=0, max_value=1, format="percent"),
                "Risk_Level": "Risk Tier",
                "Customer_Value": "Customer Value",
                "Recommended_Action": "Suggested Action",
            },
        )

elif page == "Business Insights":
    st.title("Business Insights")
    st.caption(
        "Estimates and suggested actions for the 1,409 held-out test customers. "
        "These are illustrative, not live accounts or forecasts of actual loss."
    )

    a, b, c = st.columns(3)
    a.metric("High-Risk Monthly Charges", f"{HIGH_RISK_REVENUE:,.2f}")
    b.metric("Monthly Revenue Exposure", f"{REVENUE_EXPOSURE:.1f}%")
    c.metric("Critical-Priority Customers", f"{risk_df['Priority'].eq('Critical').sum():,}")
    st.info(
        "High-risk monthly charges are the combined bills of customers whose "
        "model score is at least 70%. They are **not** expected or realized losses. "
        "The dataset does not establish a currency, so amounts are shown without a symbol."
    )

    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            st.subheader("Monthly Charges by Risk Tier")
            rev = (risk_df.groupby("Risk_Level")["MonthlyCharges"]
                   .sum().reindex(["High", "Medium", "Low"], fill_value=0)
                   .rename_axis("Risk Tier").reset_index(name="Monthly Charges"))
            chart = alt.Chart(rev).mark_bar(size=42, cornerRadiusEnd=5).encode(
                x=alt.X("Risk Tier:N", sort=["High", "Medium", "Low"], title=None),
                y=alt.Y("Monthly Charges:Q", title="Total monthly charges"),
                color=alt.Color("Risk Tier:N", scale=alt.Scale(
                    domain=["High", "Medium", "Low"], range=[CORAL, GOLD, TEAL]), legend=None),
                tooltip=["Risk Tier:N", alt.Tooltip("Monthly Charges:Q", format=",.2f")],
            ).properties(height=260)
            st.altair_chart(chart, use_container_width=True, theme="streamlit")
    with right:
        with st.container(border=True):
            st.subheader("Customers by Outreach Priority")
            priorities = (risk_df["Priority"].value_counts()
                          .reindex(["Critical", "High", "Routine"], fill_value=0)
                          .rename_axis("Priority").reset_index(name="Customers"))
            chart = alt.Chart(priorities).mark_bar(size=42, cornerRadiusEnd=5).encode(
                x=alt.X("Priority:N", sort=["Critical", "High", "Routine"], title=None),
                y=alt.Y("Customers:Q", title="Number of customers"),
                color=alt.Color("Priority:N", scale=alt.Scale(
                    domain=["Critical", "High", "Routine"], range=[CORAL, GOLD, TEAL]), legend=None),
                tooltip=["Priority:N", "Customers:Q"],
            ).properties(height=260)
            st.altair_chart(chart, use_container_width=True, theme="streamlit")

    with st.container(border=True):
        st.subheader("Retention Priority List")
        st.caption(
            f"High value = monthly charges at or above the test-set median ({VALUE_THRESHOLD:.2f}). "
            "Critical = high risk and high value. High = other high-risk customers or "
            "medium-risk, high-value customers. These are proposed business rules."
        )
        selection = st.selectbox("Outreach Priority", ["All", "Critical", "High", "Routine"])
        business = risk_df.copy()
        if selection != "All":
            business = business[business["Priority"].eq(selection)]
        priority_order = {"Critical": 0, "High": 1, "Routine": 2}
        business["Priority_Order"] = business["Priority"].map(priority_order)
        business = business.sort_values(
            ["Priority_Order", "Churn_Probability"], ascending=[True, False]
        )
        st.write(f"Showing **{len(business):,}** customers")
        st.dataframe(
            business[[
                "Demo_Row_ID", "Contract_Type", "tenure", "MonthlyCharges",
                "Churn_Probability", "Risk_Level", "Customer_Value",
                "Priority", "Recommended_Action"
            ]],
            hide_index=True, use_container_width=True,
            column_config={
                "Demo_Row_ID": "Demo Row ID",
                "Contract_Type": "Contract",
                "tenure": "Tenure (months)",
                "MonthlyCharges": st.column_config.NumberColumn("Monthly Charges", format="%.2f"),
                "Churn_Probability": st.column_config.ProgressColumn(
                    "Model Churn Score", min_value=0, max_value=1, format="percent"
                ),
                "Risk_Level": "Risk Tier",
                "Customer_Value": "Customer Value",
                "Recommended_Action": "Suggested Action",
            },
        )

else:
    st.title("Customer Drill-Down")
    st.caption("Explore an individual customer's predicted risk using the selected Phase 2 model.")
    with st.form("customer_form", border=True):
        st.subheader("Customer Information")
        a, b, c = st.columns(3)
        with a:
            gender = st.selectbox("Gender", ["Female", "Male"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])
        with b:
            senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            tenure = st.number_input("Tenure (months)", 0, 72, 12, 1)
        with c:
            partner = st.selectbox("Partner", ["No", "Yes"])
            phone = st.selectbox("Phone Service", ["No", "Yes"])

        st.subheader("Services")
        a, b, c = st.columns(3)
        with a:
            multiple = st.selectbox("Multiple Lines", ["No", "No phone service", "Yes"])
            internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            security = st.selectbox("Online Security", ["No", "No internet service", "Yes"])
        with b:
            backup = st.selectbox("Online Backup", ["No", "No internet service", "Yes"])
            protection = st.selectbox("Device Protection", ["No", "No internet service", "Yes"])
            support = st.selectbox("Tech Support", ["No", "No internet service", "Yes"])
        with c:
            tv = st.selectbox("Streaming TV", ["No", "No internet service", "Yes"])
            movies = st.selectbox("Streaming Movies", ["No", "No internet service", "Yes"])
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

        st.subheader("Billing")
        a, b, c = st.columns(3)
        with a:
            paperless = st.selectbox("Paperless Billing", ["No", "Yes"])
            monthly = st.number_input("Monthly Charges", min_value=0.0, value=70.0, step=1.0)
        with b:
            payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
            total = st.number_input("Total Charges", min_value=0.0, value=840.0, step=1.0)
        submitted = st.form_submit_button("Predict Churn", use_container_width=True)

    if submitted:
        raw = {
            "gender": gender, "SeniorCitizen": senior, "Partner": partner,
            "Dependents": dependents, "tenure": tenure, "PhoneService": phone,
            "MultipleLines": multiple, "InternetService": internet,
            "OnlineSecurity": security, "OnlineBackup": backup,
            "DeviceProtection": protection, "TechSupport": support,
            "StreamingTV": tv, "StreamingMovies": movies, "Contract": contract,
            "PaperlessBilling": paperless, "PaymentMethod": payment,
            "MonthlyCharges": monthly, "TotalCharges": total,
        }
        problems = []
        if phone == "No" and multiple != "No phone service":
            problems.append("When Phone Service is No, Multiple Lines must be 'No phone service'.")
        if phone == "Yes" and multiple == "No phone service":
            problems.append("When Phone Service is Yes, Multiple Lines cannot be 'No phone service'.")
        internet_addon_vals = [security, backup, protection, support, tv, movies]
        if internet == "No" and any(v != "No internet service" for v in internet_addon_vals):
            problems.append("With No Internet Service, every internet add-on must be 'No internet service'.")
        if internet != "No" and any(v == "No internet service" for v in internet_addon_vals):
            problems.append("With DSL or Fiber optic, internet add-ons must be Yes or No.")
        if problems:
            for issue in problems:
                st.error(issue)
        else:
            input_data = prepare_customer(raw)
            prediction = int(model.predict(input_data)[0])
            score = float(model.predict_proba(input_data)[0, 1])
            st.subheader("Prediction Result")
            if prediction:
                st.warning("The model classifies this customer as likely to churn.")
            else:
                st.success("The model classifies this customer as likely to stay.")
            st.metric("Model Churn Score", f"{score:.1%}")
            st.progress(score)
            st.write(f"**Risk Tier:** {risk_level(score)}")
            value = customer_value(monthly)
            priority = customer_priority(risk_level(score), value)
            action = retention_action(risk_level(score), priority, contract, tenure)
            with st.container(border=True):
                st.subheader("Business Guidance")
                st.write(f"**Customer Value:** {value}")
                st.write(f"**Outreach Priority:** {priority}")
                st.write(f"**Suggested Action:** {action}")
                st.caption("A rule-based suggestion, not an individualized prediction of treatment success.")
            st.caption("These are model outputs, not calibrated real-world churn probabilities. Historical data cannot establish that any retention action will prevent churn.")
