"""
View: Strategy Backtesting Lab & Performance Tearsheet
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from src.ui.components import UIComponents
from src.portfolio.backtester import QuantitativeBacktester

def render_backtest_view(ticker: str, df: pd.DataFrame):
    st.markdown("### 📊 Quantitative Strategy Backtesting Lab & Performance Tearsheet")

    # Strategy Selection Controls
    col_s1, col_s2, col_s3 = st.columns([2, 1, 1])
    with col_s1:
        strat_choice = st.selectbox(
            "Select Algorithmic Strategy to Backtest",
            [
                "ML_CONSENSUS (AI Trend & Momentum Ensemble)",
                "MA_CROSSOVER (Dual EMA 9 / EMA 21 Fast Crossover)",
                "RSI_REVERSION (Mean-Reversion Oversold/Overbought)",
                "BUY_AND_HOLD (Passive Benchmark)"
            ],
            index=0
        )
        selected_strat_code = strat_choice.split(" ")[0]
    with col_s2:
        capital = st.number_input("Starting Capital ($)", min_value=1000, max_value=10000000, value=100000, step=10000)
    with col_s3:
        st.write("")
        st.write("")
        run_btn = st.button("🚀 Run Quantitative Backtest", use_container_width=True)

    backtester = QuantitativeBacktester(initial_capital=float(capital))
    with st.spinner(f"Simulating vectorized trade executions for {ticker} using {selected_strat_code}..."):
        bt_res = backtester.run_backtest(df, strategy_type=selected_strat_code, ticker=ticker)

    s_metrics = bt_res["strategy_metrics"]
    b_metrics = bt_res["benchmark_metrics"]

    # KPI Row
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        UIComponents.render_metric_card("Strategy Return", s_metrics["Total Return"], f"CAGR: {s_metrics['CAGR']}")
    with k2:
        UIComponents.render_metric_card("Sharpe Ratio", str(s_metrics["Sharpe Ratio"]), is_positive=s_metrics["Sharpe Ratio"] >= 1.0)
    with k3:
        UIComponents.render_metric_card("Sortino Ratio", str(s_metrics["Sortino Ratio"]), is_positive=s_metrics["Sortino Ratio"] >= 1.0)
    with k4:
        UIComponents.render_metric_card("Max Drawdown", s_metrics["Max Drawdown"], is_positive=False)
    with k5:
        UIComponents.render_metric_card("Alpha vs Benchmark", s_metrics["Alpha vs Benchmark"], is_positive=float(s_metrics["Alpha vs Benchmark"].replace("%", "")) >= 0)

    st.markdown("<br/>", unsafe_allow_html=True)

    # Cumulative Growth Equity Curve
    eq_df = bt_res["equity_curve"]
    eq_fig = go.Figure()
    eq_fig.add_trace(go.Scatter(
        x=eq_df.index,
        y=eq_df["Strategy_Equity"],
        name=f"Strategy ({selected_strat_code})",
        line=dict(color="#00FFA3", width=2.5)
    ))
    eq_fig.add_trace(go.Scatter(
        x=eq_df.index,
        y=eq_df["Benchmark_Equity"],
        name=f"Benchmark ({ticker} Buy & Hold)",
        line=dict(color="#64748B", width=1.8, dash="dot")
    ))
    eq_layout = UIComponents.get_plotly_dark_layout(f"Cumulative Portfolio Growth (${capital:,.0f} Base)")
    eq_layout["height"] = 420
    eq_layout["yaxis"]["title"] = "Portfolio Equity ($)"
    eq_fig.update_layout(eq_layout)
    st.plotly_chart(eq_fig, use_container_width=True)

    # Drawdown Underwater Chart
    dd_df = bt_res["drawdown_curve"]
    dd_fig = go.Figure()
    dd_fig.add_trace(go.Scatter(
        x=dd_df.index,
        y=dd_df["Strategy_Drawdown"] * 100,
        fill="tozeroy",
        fillcolor="rgba(0, 255, 163, 0.2)",
        line=dict(color="#00FFA3", width=1.5),
        name="Strategy Drawdown"
    ))
    dd_fig.add_trace(go.Scatter(
        x=dd_df.index,
        y=dd_df["Benchmark_Drawdown"] * 100,
        line=dict(color="#FF4B4B", width=1.2, dash="dash"),
        name="Benchmark Drawdown"
    ))
    dd_layout = UIComponents.get_plotly_dark_layout("Strategy vs Benchmark Drawdown Underwater Chart")
    dd_layout["height"] = 260
    dd_layout["yaxis"]["title"] = "Drawdown (%)"
    dd_fig.update_layout(dd_layout)
    st.plotly_chart(dd_fig, use_container_width=True)

    # Detailed Tearsheet Comparison & Trade Log
    t_tab1, t_tab2, t_tab3 = st.tabs(["📋 Performance Tearsheet", "📜 Executed Trades Log", "📅 Monthly Returns"])

    with t_tab1:
        st.write("#### 🏆 Strategy vs Benchmark Performance Matrix")
        metrics_comp = []
        for k in ["Total Return", "CAGR", "Sharpe Ratio", "Sortino Ratio", "Max Drawdown", "Annualized Volatility"]:
            metrics_comp.append({
                "Performance Metric": k,
                "Strategy": s_metrics.get(k, "N/A"),
                "Buy & Hold Benchmark": b_metrics.get(k, "N/A")
            })
        st.dataframe(pd.DataFrame(metrics_comp), use_container_width=True, hide_index=True)

    with t_tab2:
        st.write("#### 📜 Executed Trade History")
        trades = bt_res.get("trade_logs", [])
        if trades:
            trades_df = pd.DataFrame(trades)
            st.dataframe(trades_df, use_container_width=True, hide_index=True)
        else:
            st.info("No closed trades recorded in this backtest window.")

    with t_tab3:
        st.write("#### 📅 Monthly Returns Summary (%)")
        m_rets = bt_res.get("monthly_returns", {})
        if m_rets:
            m_df = pd.DataFrame(list(m_rets.items()), columns=["Month", "Return (%)"]).sort_values(by="Month", ascending=False)
            st.dataframe(m_df, use_container_width=True, hide_index=True)
