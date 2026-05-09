import shap
import pandas as pd
import numpy as np
import plotly.graph_objects as go

def get_shap_values(model, X_test, feature_names, model_type="rf"):
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    return shap_values, X_test

def plot_shap_bar(shap_values, feature_names):
    mean_shap = np.abs(shap_values).mean(axis=0)
    mean_shap = mean_shap[:len(feature_names)]

    df = pd.DataFrame({
        "Feature":    feature_names,
        "Importance": mean_shap
    }).sort_values("Importance", ascending=True)

    fig = go.Figure(go.Bar(
        x=df["Importance"],
        y=df["Feature"],
        orientation="h",
        marker_color="crimson"
    ))
    fig.update_layout(
        title="SHAP Feature Importance",
        xaxis_title="Mean SHAP Value",
        yaxis_title="Feature",
        height=400
    )
    return fig

def plot_shap_single(model, single_input, feature_names, model_type="rf"):
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(single_input)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    values = np.array(shap_values).flatten()
    values = values[:len(feature_names)]

    df = pd.DataFrame({
        "Feature": list(feature_names),
        "SHAP":    list(values)
    }).sort_values("SHAP")

    colors = ["green" if v < 0 else "red" for v in df["SHAP"]]

    fig = go.Figure(go.Bar(
        x=df["SHAP"],
        y=df["Feature"],
        orientation="h",
        marker_color=colors
    ))
    fig.update_layout(
        title="SHAP Values - Red increases risk, Green decreases risk",
        xaxis_title="SHAP Value",
        yaxis_title="Feature",
        height=400
    )
    return fig