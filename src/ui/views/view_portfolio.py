"""
View: Portfolio Optimizer & Efficient Frontier
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from src.config import DEFAULT_ASSET_UNIVERSES
from src.ui.components import UIComponents
from src.data.market_loader import MarketDataLoader
from src.portfolio.optimizer import PortfolioOptimizer

def render_portfolio_view():
    st.markdown("### 💼 Multi-Asset Portfolio Optimization & Markowitz Efficient Frontier")

    # Asset Universe Selection
    col_u1, col_u2 = st.columns([1, 2])
    with col_u1:
        universe_choice = st.selectbox("Select Asset Universe Preset", list(DEFAULT_ASSET_UNIVERSES.keys()), index=0)
        default_tickers = DEFAULT_ASSET_UNIVERSES[universe_choice]
    with col_u2:
        selected_tickers = st.multiselect(
            "Configure Portfolio Assets (Minimum 2 Assets Required)",
            options=default_tickers + ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "SPY", "QQQ", "BTC-USD", "ETH-USD", "RELIANCE.NS", "TCS.NS"],
            default=default_tickers[:5] if len(default_tickers) >= 5 else default_tickers
        )

    if len(selected_tickers) < 2:
        st.warning("Please select at least 2 assets to run portfolio optimization.")
        return

    market_loader = MarketDataLoader()
    optimizer = PortfolioOptimizer()

    with st.spinner("Fetching multi-asset historical returns & computing covariance matrices..."):
        prices_df, returns_df = market_loader.fetch_multi_asset_data(selected_tickers, period="2y")
        opt_res = optimizer.optimize_portfolio(returns_df)

    max_s = opt_res["max_sharpe_portfolio"]
    min_v = opt_res["min_volatility_portfolio"]
    eq_w = opt_res["equal_weight_portfolio"]
    rp_w = opt_res["risk_parity_portfolio"]

    # KPI Comparison Row
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        UIComponents.render_metric_card("Max Sharpe Return", f"{max_s['expected_return']}%", f"Sharpe: {max_s['sharpe_ratio']}")
    with p2:
        UIComponents.render_metric_card("Max Sharpe Volatility", f"{max_s['volatility']}%")
    with p3:
        UIComponents.render_metric_card("Min Volatility Return", f"{min_v['expected_return']}%", f"Sharpe: {min_v['sharpe_ratio']}")
    with p4:
        UIComponents.render_metric_card("Min Volatility Risk", f"{min_v['volatility']}%")

    st.markdown("<br/>", unsafe_allow_html=True)

    # Plotly Efficient Frontier Interactive Graph
    mc_cloud = opt_res["monte_carlo_cloud"]
    frontier = opt_res["efficient_frontier"]

    fig = go.Figure()

    # 1. Monte Carlo Simulated Portfolios Cloud
    fig.add_trace(go.Scatter(
        x=mc_cloud["volatilities"],
        y=mc_cloud["returns"],
        mode="markers",
        marker=dict(
            size=5,
            color=mc_cloud["sharpes"],
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Sharpe Ratio", len=0.8, x=1.02)
        ),
        name="Random Portfolios"
    ))

    # 2. Efficient Frontier Curve
    fig.add_trace(go.Scatter(
        x=frontier["volatilities"],
        y=frontier["returns"],
        mode="lines",
        line=dict(color="#FF4B4B", width=3),
        name="Efficient Frontier"
    ))

    # 3. Key Portfolio Tangency Markers
    fig.add_trace(go.Scatter(
        x=[max_s["volatility"]],
        y=[max_s["expected_return"]],
        mode="markers",
        marker=dict(color="#00FFA3", size=16, symbol="star"),
        name=f"Max Sharpe (SR: {max_s['sharpe_ratio']})"
    ))

    fig.add_trace(go.Scatter(
        x=[min_v["volatility"]],
        y=[min_v["expected_return"]],
        mode="markers",
        marker=dict(color="#00B4D8", size=14, symbol="diamond"),
        name=f"Min Volatility (SR: {min_v['sharpe_ratio']})"
    ))

    fig.add_trace(go.Scatter(
        x=[eq_w["volatility"]],
        y=[eq_w["expected_return"]],
        mode="markers",
        marker=dict(color="#FFAA00", size=12, symbol="circle"),
        name=f"Equal Weight (SR: {eq_w['sharpe_ratio']})"
    ))

    ef_layout = UIComponents.get_plotly_dark_layout("Markowitz Mean-Variance Efficient Frontier")
    ef_layout["height"] = 520
    ef_layout["xaxis"]["title"] = "Annualized Volatility / Risk (%)"
    ef_layout["yaxis"]["title"] = "Annualized Expected Return (%)"
    fig.update_layout(ef_layout)
    st.plotly_chart(fig, use_container_width=True)

    # Weight Allocations & Correlation Matrix
    w_col1, w_col2 = st.columns([1, 1])

    with w_col1:
        st.write("#### ⚖️ Optimal Asset Allocation Weights")
        alloc_choice = st.radio("Allocation Strategy:", ["Max Sharpe Ratio", "Minimum Volatility", "Risk Parity (Inv Vol)", "Equal Weight"], horizontal=True)

        if alloc_choice == "Max Sharpe Ratio":
            weights_data = max_s["weights"]
        elif alloc_choice == "Minimum Volatility":
            weights_data = min_v["weights"]
        elif alloc_choice == "Risk Parity (Inv Vol)":
            weights_data = rp_w["weights"]
        else:
            weights_data = eq_w["weights"]

        # Donut Chart for Weights
        pie_fig = go.Figure(data=[go.Pie(
            labels=list(weights_data.keys()),
            values=list(weights_data.values()),
            hole=0.45
        )])
        pie_layout = UIComponents.get_plotly_dark_layout(f"{alloc_choice} Weights")
        pie_layout["height"] = 320
        pie_fig.update_layout(pie_layout)
        st.plotly_chart(pie_fig, use_container_width=True)

    with w_col2:
        st.write("#### 🔗 Multi-Asset Correlation Heatmap")
        corr_matrix = opt_res["correlation_matrix"]
        
        heatmap_fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale="RdBu_r",
            zmin=-1.0,
            zmax=1.0,
            text=corr_matrix.values.round(2),
            texttemplate="%{text}",
            colorbar=dict(len=0.8)
        ))
        hm_layout = UIComponents.get_plotly_dark_layout("Asset Correlation Matrix (2Y Daily Returns)")
        hm_layout["height"] = 320
        heatmap_fig.update_layout(hm_layout)
        st.plotly_chart(heatmap_fig, use_container_width=True)
