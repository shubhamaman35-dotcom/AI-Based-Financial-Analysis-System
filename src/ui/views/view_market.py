"""
View: Market Intelligence & Technical Screener
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from src.ui.components import UIComponents
from src.models.feature_engineering import FeatureEngineer

def render_market_view(ticker: str, df: pd.DataFrame, quote: dict, fundamentals: dict):
    st.markdown("### 📈 Market Intelligence & Real-Time Technical Screener")

    # Key Stats Row
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        curr_p = float(quote.get("current_price", df["Close"].iloc[-1]))
        pct_c = float(quote.get("pct_change", 0.0))
        UIComponents.render_metric_card("Spot Price", f"${curr_p:.2f}", f"{pct_c:+.2f}%", pct_c >= 0)
    with c2:
        h52 = float(quote.get("52w_high", curr_p * 1.15))
        l52 = float(quote.get("52w_low", curr_p * 0.85))
        UIComponents.render_metric_card("52W Range", f"${l52:.2f} - ${h52:.2f}")
    with c3:
        vol = quote.get("volume", int(df["Volume"].iloc[-1]))
        UIComponents.render_metric_card("24h Volume", f"{vol:,.0f}")
    with c4:
        raw_pe = quote.get("pe_ratio", fundamentals.get("ratios", {}).get("Trailing P/E", "N/A"))
        try:
            pe_formatted = f"{float(raw_pe):.2f}"
        except (ValueError, TypeError):
            pe_formatted = str(raw_pe)
        UIComponents.render_metric_card("Trailing P/E", pe_formatted)
    with c5:
        raw_beta = quote.get("beta", 1.15)
        try:
            beta_formatted = f"{float(raw_beta):.2f}"
        except (ValueError, TypeError):
            beta_formatted = str(raw_beta)
        UIComponents.render_metric_card("Beta (vs S&P500)", beta_formatted)

    st.markdown("<br/>", unsafe_allow_html=True)

    # Compute Indicators for Plotting
    fe = FeatureEngineer()
    df_feat = fe.compute_all_features(df)

    # Chart Controls
    col_c1, col_c2, col_c3 = st.columns([2, 2, 2])
    with col_c1:
        show_ma = st.checkbox("Show SMAs (20, 50, 200)", value=True)
    with col_c2:
        show_bb = st.checkbox("Show Bollinger Bands", value=True)
    with col_c3:
        chart_period = st.selectbox("Zoom Period", ["1mo", "3mo", "6mo", "1y", "All"], index=3)

    # Filter by period for visualization
    plot_df = df_feat.copy()
    if chart_period == "1mo":
        plot_df = plot_df.tail(22)
    elif chart_period == "3mo":
        plot_df = plot_df.tail(65)
    elif chart_period == "6mo":
        plot_df = plot_df.tail(130)
    elif chart_period == "1y":
        plot_df = plot_df.tail(252)

    # Multi-panel Plotly Interactive Chart
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.60, 0.20, 0.20],
        subplot_titles=(f"{ticker} Candlestick & Overlays", "MACD Indicator", "RSI Oscillator")
    )

    # 1. Main Candlestick
    fig.add_trace(
        go.Candlestick(
            x=plot_df.index,
            open=plot_df["Open"],
            high=plot_df["High"],
            low=plot_df["Low"],
            close=plot_df["Close"],
            name="OHLC",
            increasing_line_color="#00FFA3",
            decreasing_line_color="#FF4B4B"
        ),
        row=1, col=1
    )

    # Overlays
    if show_ma:
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["SMA_20"], name="SMA 20", line=dict(color="#38BDF8", width=1.2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["SMA_50"], name="SMA 50", line=dict(color="#FBBF24", width=1.4)), row=1, col=1)
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["SMA_200"], name="SMA 200", line=dict(color="#F43F5E", width=1.6)), row=1, col=1)

    if show_bb:
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["BB_Upper"], name="BB Upper", line=dict(color="rgba(148, 163, 184, 0.4)", dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["BB_Lower"], name="BB Lower", line=dict(color="rgba(148, 163, 184, 0.4)", dash="dot"), fill="tonexty", fillcolor="rgba(148, 163, 184, 0.05)"), row=1, col=1)

    # 2. MACD Subplot
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["MACD"], name="MACD", line=dict(color="#00FFA3", width=1.5)), row=2, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["MACD_Signal"], name="Signal", line=dict(color="#FF4B4B", width=1.2)), row=2, col=1)
    fig.add_trace(go.Bar(x=plot_df.index, y=plot_df["MACD_Hist"], name="Histogram", marker_color=plot_df["MACD_Hist"].apply(lambda x: "rgba(0, 255, 163, 0.5)" if x >= 0 else "rgba(255, 75, 75, 0.5)")), row=2, col=1)

    # 3. RSI Subplot
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["RSI"], name="RSI (14)", line=dict(color="#A855F7", width=1.5)), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="rgba(255, 75, 75, 0.5)", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="rgba(0, 255, 163, 0.5)", row=3, col=1)

    layout = UIComponents.get_plotly_dark_layout(f"{ticker} Technical Diagnostic Terminal")
    layout["xaxis_rangeslider_visible"] = False
    layout["height"] = 700
    fig.update_layout(layout)
    fig.update_xaxes(gridcolor="#1E293B")
    fig.update_yaxes(gridcolor="#1E293B")

    st.plotly_chart(fig, use_container_width=True)

    # Fundamental Financial Statements Section
    st.markdown("### 🏢 Fundamental Health & Financial Statements")
    f_tab1, f_tab2, f_tab3, f_tab4 = st.tabs(["📊 Financial Health Score", "📑 Income Statement", "⚖️ Balance Sheet", "💵 Cash Flow"])

    with f_tab1:
        h_score = fundamentals.get("health_score", 75)
        h_break = fundamentals.get("score_breakdown", {})
        col_h1, col_h2 = st.columns([1, 2])
        with col_h1:
            st.metric("Piotroski-Altman Health Score", f"{h_score} / 100")
            st.progress(h_score / 100.0)
            status_text = "STRONG FINANCIAL HEALTH" if h_score >= 70 else "MODERATE SOLVENCY" if h_score >= 45 else "WEAK / ELEVATED RISK"
            st.caption(f"Status: **{status_text}**")
        with col_h2:
            st.write("**Health Factor Breakdown:**")
            if h_break:
                break_df = pd.DataFrame(list(h_break.items()), columns=["Pillar", "Score (/25)"])
                st.dataframe(break_df, use_container_width=True, hide_index=True)

    with f_tab2:
        inc_df = fundamentals.get("income_statement")
        if inc_df is not None and not inc_df.empty:
            st.dataframe(inc_df, use_container_width=True)
        else:
            st.info("Income statement data not available.")

    with f_tab3:
        bal_df = fundamentals.get("balance_sheet")
        if bal_df is not None and not bal_df.empty:
            st.dataframe(bal_df, use_container_width=True)
        else:
            st.info("Balance sheet data not available.")

    with f_tab4:
        cf_df = fundamentals.get("cash_flow")
        if cf_df is not None and not cf_df.empty:
            st.dataframe(cf_df, use_container_width=True)
        else:
            st.info("Cash flow statement data not available.")
