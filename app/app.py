"""
AI-Based Digital Banking Fraud Detection & Prevention System
Bank Officer Dashboard (Streamlit)
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import streamlit as st
import pandas as pd
import joblib
from predict import predict_transaction, get_risk_level

st.set_page_config(
    page_title="Fraud Detection & Prevention System",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 AI-Based Digital Banking Fraud Detection & Prevention System")
st.caption("Real-time transaction risk scoring for bank officers")

# --- Sidebar Navigation ---
page = st.sidebar.radio(
    "Navigate",
    ["Executive Overview", "Transaction Checker", "Fraud Analytics", "Model Performance"]
)

# --- Load model metadata (for overview stats) ---
@st.cache_resource
def load_model_info():
    model = joblib.load("../models/fraud_model.joblib")
    features = joblib.load("../models/feature_list.joblib")
    return model, features

model, features = load_model_info()


# ==================== PAGE 1: Executive Overview ====================
if page == "Executive Overview":
    st.header("Executive Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transactions (Test Set)", "818,514")
    col2.metric("Fraud Detected", "3,765")
    col3.metric("Fraud Rate", "0.56%")
    col4.metric("Model Recall", "82.39%")

    st.markdown("---")
    st.subheader("Model Performance Summary")
    perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
    perf_col1.metric("Precision", "25.43%")
    perf_col2.metric("Recall", "82.39%")
    perf_col3.metric("F1-Score", "38.86%")
    perf_col4.metric("PR-AUC", "0.7872")

    st.info(
        "This is a portfolio/demonstration system trained on the PaySim synthetic "
        "dataset. It is not connected to real banking infrastructure and should not "
        "be considered production-ready without further validation on real transaction data."
    )


# ==================== PAGE 2: Transaction Checker ====================
elif page == "Transaction Checker":
    st.header("Transaction Risk Checker")
    st.write("Enter transaction details to get a real-time fraud risk assessment.")

    col1, col2 = st.columns(2)
    with col1:
        amount = st.number_input("Transaction Amount", min_value=0.0, value=50000.0, step=1000.0)
        transaction_type = st.selectbox("Transaction Type", ["TRANSFER", "CASH_OUT"])
        hour_of_day = st.slider("Hour of Day", 0, 23, 12)
    with col2:
        old_balance_dest = st.number_input("Recipient's Balance Before", min_value=0.0, value=0.0, step=1000.0)
        new_balance_dest = st.number_input("Recipient's Balance After", min_value=0.0, value=0.0, step=1000.0)

    if st.button("Check Transaction", type="primary"):
        transaction = {
            'amount': amount,
            'oldbalanceDest': old_balance_dest,
            'newbalanceDest': new_balance_dest,
            'type_CASH_OUT': transaction_type == "CASH_OUT",
            'type_TRANSFER': transaction_type == "TRANSFER",
            'hour_of_day': hour_of_day
        }

        result = predict_transaction(transaction)

        st.markdown("---")
        risk_colors = {"Low": "🟢", "Medium": "🟡", "High": "🔴", "Critical": "⚫"}

        st.subheader(f"{risk_colors[result['risk_level']]} Risk Level: {result['risk_level']}")
        st.metric("Fraud Probability", f"{result['fraud_probability']*100:.2f}%")
        st.write(f"**Recommended Action:** {result['action']}")

        st.markdown("**Top Contributing Factors:**")
        for reason in result['top_reasons']:
            st.write(f"- {reason}")


# ==================== PAGE 3: Fraud Analytics ====================
elif page == "Fraud Analytics":
    st.header("Fraud Analytics")

    dash_data = joblib.load("../models/dashboard_data.joblib")

    st.subheader("Fraud Rate by Transaction Type")
    type_data = pd.DataFrame({
        "Type": list(dash_data["type_fraud_counts"].keys()),
        "Fraud Count": list(dash_data["type_fraud_counts"].values())
    })
    st.bar_chart(type_data.set_index("Type"))

    st.markdown(
        "**Insight:** Fraud only occurs in `TRANSFER` and `CASH_OUT` transaction types. "
        "`TRANSFER` has a higher fraud rate (0.77%) compared to `CASH_OUT` (0.18%)."
    )

    st.markdown("---")
    st.subheader("Transaction Amount: Fraud vs Non-Fraud")
    amount_stats = pd.DataFrame(dash_data["amount_summary"]).T
    amount_stats.columns = ["Non-Fraud", "Fraud"]
    st.dataframe(amount_stats.loc[["mean", "std", "min", "25%", "50%", "75%", "max"]])

    st.markdown(
        "**Insight:** Fraudulent transactions have an average amount of "
        f"Rs. {dash_data['amount_summary']['mean'][1]:,.0f}, "
        f"which is roughly {dash_data['amount_summary']['mean'][1] / dash_data['amount_summary']['mean'][0]:.1f}x "
        "higher than genuine transactions."
    )


# ==================== PAGE 4: Model Performance ====================
elif page == "Model Performance":
    st.header("Model Performance")

    dash_data = joblib.load("../models/dashboard_data.joblib")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Confusion Matrix")
        cm = dash_data["confusion_matrix"]
        cm_df = pd.DataFrame(
            cm,
            index=["Actual: No Fraud", "Actual: Fraud"],
            columns=["Predicted: No Fraud", "Predicted: Fraud"]
        )
        st.dataframe(cm_df)

    with col2:
        st.subheader("Precision-Recall Curve")
        pr_df = pd.DataFrame({
            "Recall": dash_data["recall_vals"][::20],
            "Precision": dash_data["precision_vals"][::20]
        })
        st.line_chart(pr_df.set_index("Recall"))

    st.markdown("---")
    st.subheader("Global Feature Importance (SHAP)")
    st.image(
        "../models/shap_summary.png",
        caption="SHAP Summary Plot — shows which features most influence fraud predictions across all transactions"
    )

    st.warning(
        "Feature importance shows correlation with the model's predictions, not causation. "
        "These are statistical patterns learned from historical data, not proven cause-and-effect relationships."
    )