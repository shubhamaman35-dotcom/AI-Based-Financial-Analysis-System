"""
View: AI Deep Learning Forecasting & Explainable AI (XAI)
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from src.ui.components import UIComponents
from src.models.deep_learning_model import DeepLearningForecaster
from src.models.ml_classifiers import SignalClassifier
from src.xai.explainability import ModelExplainer
from src.models.feature_engineering import FeatureEngineer

def render_forecast_view(ticker: str, df: pd.DataFrame, verdict_data: dict):
    st.markdown("### 🧠 AI Deep Learning Forecasting & Model Interpretability")

    # Forecast Horizon Selection
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        horizon_days = st.selectbox("Forecast Horizon", [7, 14, 30, 60], index=2)
    with col2:
        epochs = st.slider("Neural Network Epochs", min_value=10, max_value=60, value=30, step=10)
    with col3:
        st.write("")
        st.write("")
        train_btn = st.button("⚡ Re-train Neural Forecaster", use_container_width=True)

    # Train Forecaster
    with st.spinner(f"Training Deep Sequence Model for {ticker}..."):
        forecaster = DeepLearningForecaster(seq_length=25, hidden_dim=64, epochs=epochs)
        forecast_res = forecaster.train_and_forecast(df, target_col="Close", forecast_days=horizon_days)

    # Train Classifier for XAI
    clf = SignalClassifier()
    clf_res = clf.train_and_evaluate(df)

    # Key Metrics Row
    m1, m2, m3, m4, m5 = st.columns(5)
    metrics = forecast_res["metrics"]
    with m1:
        UIComponents.render_metric_card("Expected Return", f"{forecast_res['expected_return_30d']}%", is_positive=forecast_res['expected_return_30d'] >= 0)
    with m2:
        UIComponents.render_metric_card("Model RMSE", f"${metrics['RMSE']}")
    with m3:
        UIComponents.render_metric_card("Model MAE", f"${metrics['MAE']}")
    with m4:
        UIComponents.render_metric_card("MAPE Error", metrics["MAPE"])
    with m5:
        UIComponents.render_metric_card("Directional Acc", metrics["Directional_Accuracy"])

    st.markdown("<br/>", unsafe_allow_html=True)

    # Deep Learning Forecast Chart with Confidence Intervals
    forecast_df = forecast_res["forecast_df"]
    hist_tail = df["Close"].tail(90)

    fig = go.Figure()

    # Historical Price
    fig.add_trace(go.Scatter(
        x=hist_tail.index,
        y=hist_tail.values,
        name="Historical Price",
        line=dict(color="#38BDF8", width=2.5)
    ))

    # Upper and Lower 95% Confidence Band
    fig.add_trace(go.Scatter(
        x=forecast_df.index,
        y=forecast_df["Upper_95"],
        name="Upper 95% Bound",
        line=dict(color="rgba(0, 255, 163, 0.15)", width=0),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=forecast_df.index,
        y=forecast_df["Lower_95"],
        name="95% Confidence Band",
        line=dict(color="rgba(0, 255, 163, 0.15)", width=0),
        fill="tonexty",
        fillcolor="rgba(0, 255, 163, 0.10)"
    ))

    # Upper and Lower 80% Confidence Band
    fig.add_trace(go.Scatter(
        x=forecast_df.index,
        y=forecast_df["Upper_80"],
        name="Upper 80% Bound",
        line=dict(color="rgba(0, 255, 163, 0.25)", width=0),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=forecast_df.index,
        y=forecast_df["Lower_80"],
        name="80% Confidence Band",
        line=dict(color="rgba(0, 255, 163, 0.25)", width=0),
        fill="tonexty",
        fillcolor="rgba(0, 255, 163, 0.18)"
    ))

    # Point Forecast
    fig.add_trace(go.Scatter(
        x=forecast_df.index,
        y=forecast_df["Forecast"],
        name="AI Point Forecast",
        line=dict(color="#00FFA3", width=3, dash="solid")
    ))

    layout = UIComponents.get_plotly_dark_layout(f"{ticker} AI Multi-Horizon Trajectory Projection ({horizon_days} Days)")
    layout["height"] = 520
    fig.update_layout(layout)
    st.plotly_chart(fig, use_container_width=True)

    # Diagnostic Sub-Tabs
    d_tab1, d_tab2, d_tab3 = st.tabs(["🔍 Explainable AI (XAI)", "📉 Training Loss Curve", "🎯 Test Set Validation"])

    with d_tab1:
        explainer = ModelExplainer()
        fe = FeatureEngineer()
        latest_feat = fe.compute_all_features(df).iloc[-1]
        xai_res = explainer.explain_prediction(clf_res["feature_importance"], latest_feat, verdict_data)

        st.info(xai_res["narrative_explanation"])

        c_x1, c_x2 = st.columns([1, 1])
        with c_x1:
            st.write("**Top Alpha Drivers & Indicator Attribution:**")
            st.dataframe(xai_res["top_drivers"][["Feature", "Importance", "Current_Value", "Impact", "Note"]], use_container_width=True, hide_index=True)
        with c_x2:
            # Bar chart of feature importance
            top_df = clf_res["feature_importance"].head(10)
            bar_fig = go.Figure(go.Bar(
                x=top_df["Importance"],
                y=top_df["Feature"],
                orientation="h",
                marker=dict(color="#00FFA3")
            ))
            bar_layout = UIComponents.get_plotly_dark_layout("Feature Importance Breakdown (Random Forest & Gradient Boosting)")
            bar_layout["height"] = 350
            bar_fig.update_layout(bar_layout)
            bar_fig.update_yaxes(autorange="reversed")
            st.plotly_chart(bar_fig, use_container_width=True)

    with d_tab2:
        loss_hist = forecast_res["loss_history"]
        loss_fig = go.Figure()
        loss_fig.add_trace(go.Scatter(
            x=list(range(1, len(loss_hist) + 1)),
            y=loss_hist,
            name="MSE Loss",
            line=dict(color="#F59E0B", width=2)
        ))
        loss_layout = UIComponents.get_plotly_dark_layout("Neural Network Adam Optimizer Convergence (Loss vs Epochs)")
        loss_layout["height"] = 350
        loss_layout["xaxis"]["title"] = "Epoch"
        loss_layout["yaxis"]["title"] = "Mean Squared Error"
        loss_fig.update_layout(loss_layout)
        st.plotly_chart(loss_fig, use_container_width=True)

    with d_tab3:
        test_comp = forecast_res["test_comparison"]
        val_fig = go.Figure()
        val_fig.add_trace(go.Scatter(x=test_comp.index, y=test_comp["Actual"], name="Actual Price", line=dict(color="#94A3B8", width=2)))
        val_fig.add_trace(go.Scatter(x=test_comp.index, y=test_comp["Predicted"], name="Model Predicted", line=dict(color="#00FFA3", width=2, dash="dot")))
        val_layout = UIComponents.get_plotly_dark_layout("Out-of-Sample Test Set Validation (Actual vs Predicted)")
        val_layout["height"] = 350
        val_fig.update_layout(val_layout)
        st.plotly_chart(val_fig, use_container_width=True)
