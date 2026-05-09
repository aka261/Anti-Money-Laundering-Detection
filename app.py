import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import math

from data.data_generator import generate_aml_dataset
from models.trainer import train_all_models
from utils.explainer import explain_transaction, explain_model_results
from utils.shap_explainer import get_shap_values, plot_shap_bar, plot_shap_single

st.set_page_config(page_title="AML Detection System", layout="wide")
st.title("Anti-Money Laundering Detection System")

# Train button instead of auto cache
if "trained" not in st.session_state:
    st.session_state.trained = False

if not st.session_state.trained:
    st.info("Click the button below to generate data and train models.")
    if st.button("Generate Data and Train Models"):
        with st.spinner("Generating dataset and training models. Please wait 5 minutes..."):
            df = generate_aml_dataset(n_samples=50000)
            results, scaler, rf, xgb, nn, X_test, y_test, feature_cols = train_all_models(df)
            st.session_state.df           = df
            st.session_state.results      = results
            st.session_state.scaler       = scaler
            st.session_state.rf_model     = rf
            st.session_state.xgb_model    = xgb
            st.session_state.nn_model     = nn
            st.session_state.X_test       = X_test
            st.session_state.y_test       = y_test
            st.session_state.feature_cols = feature_cols
            st.session_state.trained      = True
            st.rerun()
    st.stop()

df           = st.session_state.df
results      = st.session_state.results
scaler       = st.session_state.scaler
rf_model     = st.session_state.rf_model
xgb_model    = st.session_state.xgb_model
nn_model     = st.session_state.nn_model
X_test       = st.session_state.X_test
y_test       = st.session_state.y_test
feature_cols = st.session_state.feature_cols

with st.sidebar:
    st.header("Navigation")
    page = st.radio("Go to", [
        "Dataset Overview",
        "Model Comparison",
        "Transaction Risk Checker",
        "SHAP Explainability",
        "Suspicious Transactions",
        "MLflow Tracking"
    ])
    if st.button("Retrain Models"):
        st.session_state.trained = False
        st.rerun()

# ── Page 1: Dataset Overview ───────────────────────────────────────────────────
if page == "Dataset Overview":
    st.header("Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transactions",      f"{len(df):,}")
    col2.metric("Normal Transactions",     f"{(df['is_laundering']==0).sum():,}")
    col3.metric("Suspicious Transactions", f"{(df['is_laundering']==1).sum():,}")
    col4.metric("Fraud Rate",              f"{df['is_laundering'].mean()*100:.1f}%")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(values=df["is_laundering"].value_counts().values,
                     names=["Normal", "Suspicious"],
                     title="Transaction Distribution")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.histogram(df, x="amount", color="is_laundering", nbins=50,
                           title="Transaction Amount Distribution",
                           labels={"is_laundering": "Suspicious"})
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x="hour", color="is_laundering",
                           title="Transactions by Hour of Day",
                           labels={"is_laundering": "Suspicious"})
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.histogram(df, x="transaction_type", color="is_laundering",
                           title="Transactions by Type",
                           labels={"is_laundering": "Suspicious"})
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Sample Data")
    st.dataframe(df.head(20))

# ── Page 2: Model Comparison ───────────────────────────────────────────────────
elif page == "Model Comparison":
    st.header("Model Comparison")

    col1, col2, col3 = st.columns(3)
    for i, model_name in enumerate(results.keys()):
        col = [col1, col2, col3][i]
        with col:
            st.subheader(model_name)
            st.metric("Accuracy",  f"{results[model_name]['accuracy']*100:.1f}%")
            st.metric("Precision", f"{results[model_name]['precision']*100:.1f}%")
            st.metric("Recall",    f"{results[model_name]['recall']*100:.1f}%")
            st.metric("F1 Score",  f"{results[model_name]['f1']*100:.1f}%")
            st.metric("AUC Score", f"{results[model_name]['auc']:.3f}")

    metrics_df = pd.DataFrame(results).T.reset_index()
    metrics_df.columns = ["Model", "Accuracy", "Precision", "Recall", "F1", "AUC", "Confusion Matrix"]
    metrics_df = metrics_df.drop(columns=["Confusion Matrix"])
    plot_df    = metrics_df.melt(id_vars=["Model"], var_name="Metric", value_name="Score")
    fig = px.bar(plot_df, x="Metric", y="Score", color="Model",
                 barmode="group", title="Model Performance Comparison")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Confusion Matrices")
    col1, col2, col3 = st.columns(3)
    for i, (model_name, result) in enumerate(results.items()):
        col = [col1, col2, col3][i]
        with col:
            cm  = result["confusion_matrix"]
            fig = px.imshow(cm, text_auto=True,
                            labels=dict(x="Predicted", y="Actual"),
                            x=["Normal", "Suspicious"],
                            y=["Normal", "Suspicious"],
                            title=f"{model_name}")
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("AI Explanation")
    if st.button("Explain Results"):
        with st.spinner("Generating explanation..."):
            explanation = explain_model_results(results)
            st.write(explanation)

# ── Page 3: Transaction Risk Checker ──────────────────────────────────────────
elif page == "Transaction Risk Checker":
    st.header("Transaction Risk Checker")

    col1, col2 = st.columns(2)
    with col1:
        amount           = st.number_input("Transaction Amount ($)", min_value=1.0, value=5000.0)
        transaction_type = st.selectbox("Transaction Type", ["transfer", "payment", "deposit", "withdrawal"])
        hour             = st.slider("Hour of Day", 0, 23, 10)
        day_of_week      = st.selectbox("Day of Week", [0,1,2,3,4,5,6],
                                        format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x])
    with col2:
        num_tx_sender   = st.number_input("Number of Transactions by Sender Today", min_value=1, value=3)
        num_tx_receiver = st.number_input("Number of Transactions by Receiver Today", min_value=1, value=3)
        same_bank       = st.selectbox("Same Bank Transaction?", [1, 0], format_func=lambda x: "Yes" if x else "No")
        international   = st.selectbox("International Transaction?", [0, 1], format_func=lambda x: "Yes" if x else "No")
        model_choice    = st.selectbox("Model to Use", ["Random Forest", "XGBoost", "Neural Network"])

    if st.button("Check Transaction"):
        transaction = {
            "amount":                      amount,
            "amount_log":                  math.log1p(amount),
            "hour":                        hour,
            "day_of_week":                 day_of_week,
            "num_transactions_sender":     num_tx_sender,
            "num_transactions_receiver":   num_tx_receiver,
            "same_bank":                   same_bank,
            "international":               international,
            "transaction_type_payment":    1 if transaction_type == "payment"    else 0,
            "transaction_type_transfer":   1 if transaction_type == "transfer"   else 0,
            "transaction_type_withdrawal": 1 if transaction_type == "withdrawal" else 0,
        }

        input_df     = pd.DataFrame([transaction])
        input_scaled = scaler.transform(input_df)

        if model_choice == "Random Forest":
            risk_score = rf_model.predict_proba(input_scaled)[0][1]
        elif model_choice == "XGBoost":
            risk_score = xgb_model.predict_proba(input_scaled)[0][1]
        else:
            risk_score = float(nn_model.predict(input_scaled)[0][0])

        if risk_score >= 0.7:
            st.error(f"HIGH RISK - Risk Score: {risk_score:.2f}")
        elif risk_score >= 0.4:
            st.warning(f"MEDIUM RISK - Risk Score: {risk_score:.2f}")
        else:
            st.success(f"LOW RISK - Risk Score: {risk_score:.2f}")

        fig = go.Figure(go.Indicator(
            mode  = "gauge+number",
            value = risk_score * 100,
            title = {"text": "Risk Score"},
            gauge = {
                "axis":  {"range": [0, 100]},
                "bar":   {"color": "darkred"},
                "steps": [
                    {"range": [0,  40], "color": "green"},
                    {"range": [40, 70], "color": "yellow"},
                    {"range": [70, 100], "color": "red"},
                ]
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("AI Explanation")
        with st.spinner("Analyzing transaction..."):
            explanation = explain_transaction(transaction, risk_score)
            st.write(explanation)

# ── Page 4: SHAP Explainability ────────────────────────────────────────────────
elif page == "SHAP Explainability":
    st.header("SHAP Explainability")
    st.write("SHAP shows which features are driving the model predictions.")

    model_choice = st.selectbox("Select Model", ["Random Forest", "XGBoost"])
    model        = rf_model if model_choice == "Random Forest" else xgb_model

    if st.button("Generate SHAP Values"):
        with st.spinner("Calculating SHAP values. This may take a minute..."):
            sample_size         = min(500, len(X_test))
            X_sample            = X_test[:sample_size]
            shap_values, X_used = get_shap_values(model, X_sample, feature_cols, "rf")
            fig                 = plot_shap_bar(shap_values, feature_cols)
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Single Transaction SHAP")
    col1, col2 = st.columns(2)
    with col1:
        amount2          = st.number_input("Amount", min_value=1.0, value=9500.0, key="shap_amount")
        transaction_type2= st.selectbox("Type", ["transfer", "payment", "deposit", "withdrawal"], key="shap_type")
        hour2            = st.slider("Hour", 0, 23, 2, key="shap_hour")
        day2             = st.selectbox("Day", [0,1,2,3,4,5,6], index=6,
                                        format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x],
                                        key="shap_day")
    with col2:
        sender2   = st.number_input("Sender Transactions",   min_value=1, value=50, key="shap_sender")
        receiver2 = st.number_input("Receiver Transactions", min_value=1, value=45, key="shap_receiver")
        bank2     = st.selectbox("Same Bank?",    [1, 0], format_func=lambda x: "Yes" if x else "No", key="shap_bank")
        intl2     = st.selectbox("International?",[0, 1], format_func=lambda x: "Yes" if x else "No", key="shap_intl")

    if st.button("Explain this Transaction"):
        transaction = {
            "amount":                      amount2,
            "amount_log":                  math.log1p(amount2),
            "hour":                        hour2,
            "day_of_week":                 day2,
            "num_transactions_sender":     sender2,
            "num_transactions_receiver":   receiver2,
            "same_bank":                   bank2,
            "international":               intl2,
            "transaction_type_payment":    1 if transaction_type2 == "payment"    else 0,
            "transaction_type_transfer":   1 if transaction_type2 == "transfer"   else 0,
            "transaction_type_withdrawal": 1 if transaction_type2 == "withdrawal" else 0,
        }
        input_df     = pd.DataFrame([transaction])
        input_scaled = scaler.transform(input_df)
        model        = rf_model if model_choice == "Random Forest" else xgb_model

        with st.spinner("Calculating SHAP values..."):
            fig = plot_shap_single(model, input_scaled, feature_cols, "rf")
            st.plotly_chart(fig, use_container_width=True)
            st.info("Red bars increase the risk score. Green bars decrease the risk score.")

# ── Page 5: Suspicious Transactions ───────────────────────────────────────────
elif page == "Suspicious Transactions":
    st.header("Suspicious Transactions")
    suspicious = df[df["is_laundering"] == 1].head(100)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(suspicious, x="amount", y="num_transactions_sender",
                         color="transaction_type", title="Suspicious Transaction Patterns")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.histogram(suspicious, x="amount", nbins=30,
                           title="Suspicious Transaction Amounts")
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(suspicious[[
        "transaction_id", "sender_account", "receiver_account",
        "amount", "transaction_type", "hour", "international"
    ]])

# ── Page 6: MLflow Tracking ────────────────────────────────────────────────────
elif page == "MLflow Tracking":
    st.header("MLflow Experiment Tracking")
    st.code("mlflow ui", language="bash")
    st.write("Run the above command in your terminal then open http://localhost:5000")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Parameters**")
        st.write("- Model type")
        st.write("- n_estimators")
        st.write("- epochs")
    with col2:
        st.markdown("**Metrics**")
        st.write("- Accuracy")
        st.write("- Precision")
        st.write("- Recall")
        st.write("- F1 Score")
        st.write("- AUC Score")
    with col3:
        st.markdown("**Artifacts**")
        st.write("- Trained model files")
        st.write("- Run history")

    for model_name, result in results.items():
        with st.expander(model_name):
            st.write(f"Accuracy  : {result['accuracy']*100:.1f}%")
            st.write(f"Precision : {result['precision']*100:.1f}%")
            st.write(f"Recall    : {result['recall']*100:.1f}%")
            st.write(f"F1 Score  : {result['f1']*100:.1f}%")
            st.write(f"AUC Score : {result['auc']:.3f}")