"""
View: Paper Trading & Virtual Portfolio Manager
Enhanced with Stop-Loss (SL), Take-Profit (TP), Order Book filters, and Live P&L color coding.
"""

import streamlit as st
import pandas as pd

from src.ui.components import UIComponents
from src.portfolio.paper_trader import PaperTradingWallet

def render_paper_trading_view(ticker: str, current_price: float):
    st.markdown("### 💰 Real-Time Paper Trading & Virtual Portfolio Manager")

    wallet = PaperTradingWallet()
    
    # Portfolio summary given current ticker price
    current_prices = {ticker: current_price}
    
    # Check and trigger any automated Stop-Loss or Take-Profit orders
    triggered_orders = wallet.check_and_trigger_risk_orders(current_prices)
    for trg in triggered_orders:
        st.warning(f"🚨 Risk Trigger: {trg['type']} executed on {trg['ticker']} at ${trg['price']:.2f}")

    summary = wallet.get_portfolio_summary(current_prices)

    # Top KPI Metrics Row
    w1, w2, w3, w4, w5 = st.columns(5)
    with w1:
        UIComponents.render_metric_card("Total Portfolio", f"${summary['total_portfolio_value']:,.2f}")
    with w2:
        UIComponents.render_metric_card("Cash Balance", f"${summary['cash']:,.2f}")
    with w3:
        UIComponents.render_metric_card("Holdings Value", f"${summary['holdings_value']:,.2f}")
    with w4:
        pnl_val = summary['total_pnl']
        pnl_pct = summary['total_pnl_pct']
        UIComponents.render_metric_card("Total P&L", f"${pnl_val:,.2f}", f"{pnl_pct:+.2f}%", is_positive=pnl_val >= 0)
    with w5:
        r_pnl = summary['realized_pnl']
        UIComponents.render_metric_card("Realized P&L", f"${r_pnl:,.2f}", is_positive=r_pnl >= 0)

    st.markdown("<br/>", unsafe_allow_html=True)

    # Order Execution Panel with Risk Management (SL / TP)
    st.markdown("#### ⚡ Order Execution Panel & Risk Management Controls")
    
    col_type, col_action, col_shares, col_sl, col_tp = st.columns(5)
    with col_type:
        order_type = st.selectbox("Order Type", ["MARKET", "LIMIT"])
    with col_action:
        order_action = st.selectbox("Action", ["BUY", "SELL"])
    with col_shares:
        order_shares = st.number_input(f"Quantity ({ticker})", min_value=1, max_value=10000, value=10, step=5)
    with col_sl:
        default_sl = round(current_price * 0.95, 2) if order_action == "BUY" else round(current_price * 1.05, 2)
        use_sl = st.checkbox("Enable Stop-Loss", value=False)
        stop_loss_val = st.number_input("Stop-Loss ($)", min_value=0.0, value=float(default_sl), step=0.5, format="%.2f") if use_sl else None
    with col_tp:
        default_tp = round(current_price * 1.10, 2) if order_action == "BUY" else round(current_price * 0.90, 2)
        use_tp = st.checkbox("Enable Take-Profit", value=False)
        take_profit_val = st.number_input("Take-Profit ($)", min_value=0.0, value=float(default_tp), step=0.5, format="%.2f") if use_tp else None

    # Summary & Confirm
    est_total = order_shares * current_price
    c_info, c_btn = st.columns([3, 1])
    with c_info:
        sl_text = f"${stop_loss_val:.2f}" if stop_loss_val else "None"
        tp_text = f"${take_profit_val:.2f}" if take_profit_val else "None"
        st.markdown(
            f"""
            <div style="background-color: #151922; padding: 10px 16px; border-radius: 8px; border: 1px solid #1E293B; font-size: 0.9rem;">
                <b>Order Preview:</b> {order_action} {order_shares} {ticker} @ ${current_price:,.2f} | Est: <b>${est_total:,.2f}</b> | SL: <span style="color: #FF4B4B;">{sl_text}</span> | TP: <span style="color: #00FFA3;">{tp_text}</span>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c_btn:
        st.write("")
        submit_order = st.button(f"Confirm {order_action} Order", use_container_width=True)

    if submit_order:
        res = wallet.execute_order(
            ticker=ticker,
            action=order_action,
            shares=float(order_shares),
            current_price=current_price,
            stop_loss=stop_loss_val,
            take_profit=take_profit_val,
            order_type=order_type
        )
        if res["success"]:
            st.success(res["message"])
            st.rerun()
        else:
            st.error(res["message"])

    st.markdown("---")

    # Positions and History Tabs
    p_tab1, p_tab2, p_tab3 = st.tabs(["📦 Active Positions Table", "📜 Order Book & History", "⚙️ Wallet Settings"])

    with p_tab1:
        st.write("#### 📦 Active Positions Table (Color-Coded Live P&L)")
        pos_table = summary.get("positions_table", [])
        if pos_table:
            df_pos = pd.DataFrame(pos_table)
            
            # Display formatted columns
            st.dataframe(
                df_pos[["Ticker", "Shares", "Avg Entry Price", "Current Price", "Market Value", "Stop Loss", "Take Profit", "Unrealized P&L"]],
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No open stock positions currently held. Virtual portfolio is 100% in cash.")

    with p_tab2:
        st.write("#### 📜 Order Book & Execution History Log")
        
        # Filter buttons
        filter_status = st.radio("Filter Status:", ["ALL", "EXECUTED", "PENDING", "CANCELED"], horizontal=True)
        orders = summary.get("order_history", [])
        
        if orders:
            df_orders = pd.DataFrame(orders)
            if filter_status != "ALL" and "status" in df_orders.columns:
                df_orders = df_orders[df_orders["status"] == filter_status]
                
            st.dataframe(df_orders, use_container_width=True, hide_index=True)
        else:
            st.info("No trade orders executed yet.")

    with p_tab3:
        st.write("#### ⚙️ Reset Virtual Paper Trading Wallet")
        st.caption("Resets cash balance back to $100,000 and clears all positions & history.")
        if st.button("⚠️ Reset Paper Wallet to $100,000"):
            wallet.reset_wallet()
            st.success("Paper wallet successfully reset!")
            st.rerun()
