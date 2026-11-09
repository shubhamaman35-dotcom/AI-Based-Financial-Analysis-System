"""
View: Quantitative Risk & Monte Carlo Simulation
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from src.ui.components import UIComponents
from src.risk.monte_carlo import MonteCarloSimulator
from src.risk.risk_metrics import RiskMetricsEngine

def render_risk_view(ticker: str, df: pd.DataFrame):
    st.markdown("### 🎲 Quantitative Risk Engine & Stochastic Monte Carlo Simulation")

    risk_engine = RiskMetricsEngine()
    risk_data = risk_engine.compute_all_risk_metrics(df, ticker=ticker)

    # Top KPI Risk Metrics
    r1, r2, r3, r4, r5 = st.columns(5)
    with r1:
        UIComponents.render_metric_card("Annualized Vol", f"{float(risk_data['annualized_volatility']):.2f}%")
    with r2:
        UIComponents.render_metric_card("Sharpe Ratio", f"{float(risk_data['sharpe_ratio']):.2f}", is_positive=risk_data["sharpe_ratio"] >= 1.0)
    with r3:
        UIComponents.render_metric_card("Sortino (Downside)", f"{float(risk_data['sortino_ratio']):.2f}", is_positive=risk_data["sortino_ratio"] >= 1.0)
    with r4:
        UIComponents.render_metric_card("30D 95% VaR", f"-{float(risk_data['var_95_30d']):.2f}%", is_positive=False)
    with r5:
        UIComponents.render_metric_card("Max Drawdown", f"{float(risk_data['max_drawdown']):.2f}%", is_positive=False)

    st.markdown("<br/>", unsafe_allow_html=True)

    # Monte Carlo Controls
    mc_col1, mc_col2, mc_col3 = st.columns([1, 1, 1])
    with mc_col1:
        mc_days = st.selectbox("Simulation Horizon", [15, 30, 60, 90, 180], index=1)
    with mc_col2:
        mc_sims = st.selectbox("Number of Paths", [250, 500, 1000, 2000], index=2)
    with mc_col3:
        mc_model = st.selectbox("Stochastic Process", ["GBM_JUMP (Merton Jump Diffusion)", "GBM (Pure Brownian Motion)"], index=0)
        selected_model = "GBM_JUMP" if "JUMP" in mc_model else "GBM"

    simulator = MonteCarloSimulator(num_simulations=mc_sims, days=mc_days)
    with st.spinner(f"Simulating {mc_sims:,} stochastic price trajectories for {ticker}..."):
        mc_res = simulator.run_simulation(df, days=mc_days, num_simulations=mc_sims, model_type=selected_model)

    # Monte Carlo Multi-Path Plotly Chart
    q_df = mc_res["quantile_df"]
    sample_paths = mc_res["paths_sample"]

    fig = go.Figure()

    # Plot sample paths (faded background lines)
    for col in sample_paths.columns[:35]:
        fig.add_trace(go.Scatter(
            x=sample_paths.index,
            y=sample_paths[col],
            mode="lines",
            line=dict(color="rgba(56, 189, 248, 0.08)", width=1),
            showlegend=False
        ))

    # Add 5th and 95th percentile confidence envelope
    fig.add_trace(go.Scatter(
        x=q_df.index,
        y=q_df["P_95"],
        name="95th Percentile (Optimistic)",
        line=dict(color="#00FFA3", width=2.5, dash="dash")
    ))
    fig.add_trace(go.Scatter(
        x=q_df.index,
        y=q_df["P_50"],
        name="50th Percentile (Median Path)",
        line=dict(color="#FFFFFF", width=3)
    ))
    fig.add_trace(go.Scatter(
        x=q_df.index,
        y=q_df["P_5"],
        name="5th Percentile (Severe Bearish)",
        line=dict(color="#FF4B4B", width=2.5, dash="dash")
    ))

    mc_layout = UIComponents.get_plotly_dark_layout(f"{ticker} Monte Carlo Simulation ({mc_sims:,} Paths & Quantile Envelopes)")
    mc_layout["height"] = 480
    fig.update_layout(mc_layout)
    st.plotly_chart(fig, use_container_width=True)

    # Probability Outcomes & Terminal Price Distribution
    p_col1, p_col2 = st.columns([1, 1])

    with p_col1:
        st.write("#### 🎯 Simulated Probability Outcomes")
        o1, o2, o3 = st.columns(3)
        with o1:
            st.metric("Probability of Profit", f"{mc_res['prob_profit']}%")
        with o2:
            st.metric("Prob of Gain > 10%", f"{mc_res['prob_gain_10pct']}%")
        with o3:
            st.metric("Prob of Loss > 10%", f"{mc_res['prob_loss_10pct']}%")

        # Distribution Histogram
        term_prices = mc_res["terminal_prices"]
        hist_fig = go.Figure(data=[go.Histogram(
            x=term_prices,
            nbinsx=40,
            marker_color="#38BDF8",
            opacity=0.8
        )])
        hist_fig.add_vline(x=mc_res["current_price"], line_dash="dash", line_color="#FFAA00", annotation_text="Spot Price")
        h_layout = UIComponents.get_plotly_dark_layout("Terminal Price Distribution at Horizon End")
        h_layout["height"] = 280
        hist_fig.update_layout(h_layout)
        st.plotly_chart(hist_fig, use_container_width=True)

    with p_col2:
        st.write("#### ⚡ Historical Market Crisis Stress Testing")
        st.caption("Simulates severe macro liquidity shocks and crash impacts on current spot value:")
        stress_df = pd.DataFrame(risk_data["stress_tests"])
        st.dataframe(stress_df, use_container_width=True, hide_index=True)

        # Historical Drawdown Chart
        dd_series = risk_data["drawdown_series"].tail(252)
        dd_fig = go.Figure()
        dd_fig.add_trace(go.Scatter(
            x=dd_series.index,
            y=dd_series.values * 100,
            fill="tozeroy",
            fillcolor="rgba(255, 75, 75, 0.25)",
            line=dict(color="#FF4B4B", width=1.5),
            name="Drawdown %"
        ))
        dd_layout = UIComponents.get_plotly_dark_layout("Historical Drawdown Underwater Profile (1 Year)")
        dd_layout["height"] = 230
        dd_layout["yaxis"]["title"] = "Drawdown (%)"
        dd_fig.update_layout(dd_layout)
        st.plotly_chart(dd_fig, use_container_width=True)
