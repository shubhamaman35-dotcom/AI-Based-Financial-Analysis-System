"""
Paper Trading Simulation Module with Advanced Risk Management
Supports Market/Limit Orders, Stop-Loss (SL), Take-Profit (TP), and Order Status tracking.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from src.config import DATA_CACHE_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PaperTradingWallet:
    """
    Simulated Paper Trading Wallet with Stop-Loss, Take-Profit, and Order History.
    """

    def __init__(self, wallet_file: Path = DATA_CACHE_DIR / "paper_wallet.json", initial_capital: float = 100000.0):
        self.wallet_file = wallet_file
        self.initial_capital = initial_capital
        self.state = self._load_wallet()

    def _load_wallet(self) -> Dict[str, Any]:
        if self.wallet_file.exists():
            try:
                with open(self.wallet_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load paper wallet: {e}")

        default_state = {
            "cash": self.initial_capital,
            "portfolio_value": self.initial_capital,
            "realized_pnl": 0.0,
            "positions": {}, # ticker -> {"shares": float, "avg_entry_price": float, "stop_loss": float, "take_profit": float}
            "order_history": [],
            "pending_orders": []
        }
        self._save_wallet(default_state)
        return default_state

    def _save_wallet(self, state: Dict[str, Any] = None):
        if state is None:
            state = self.state
        try:
            with open(self.wallet_file, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save paper wallet: {e}")

    def execute_order(
        self,
        ticker: str,
        action: str,
        shares: float,
        current_price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        order_type: str = "MARKET"
    ) -> Dict[str, Any]:
        """
        Executes BUY or SELL order with Stop-Loss (SL) and Take-Profit (TP) risk buffers.
        """
        action = action.upper()
        if shares <= 0:
            return {"success": False, "message": "Shares must be greater than 0."}

        order_cost = shares * current_price
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Validate Stop-Loss and Take-Profit logic
        if stop_loss is not None and stop_loss > 0:
            if action == "BUY" and stop_loss >= current_price:
                return {"success": False, "message": f"Stop-Loss (${stop_loss}) must be below current entry price (${current_price})."}
            if action == "SELL" and stop_loss <= current_price:
                return {"success": False, "message": f"Stop-Loss (${stop_loss}) must be above current entry price (${current_price})."}

        if take_profit is not None and take_profit > 0:
            if action == "BUY" and take_profit <= current_price:
                return {"success": False, "message": f"Take-Profit (${take_profit}) must be above current entry price (${current_price})."}
            if action == "SELL" and take_profit >= current_price:
                return {"success": False, "message": f"Take-Profit (${take_profit}) must be below current entry price (${current_price})."}

        if action == "BUY":
            if self.state["cash"] < order_cost:
                return {"success": False, "message": f"Insufficient cash (${round(self.state['cash'], 2):,.2f}) for order (${round(order_cost, 2):,.2f})."}

            self.state["cash"] -= order_cost
            if ticker in self.state["positions"]:
                curr_shares = self.state["positions"][ticker]["shares"]
                curr_avg = self.state["positions"][ticker]["avg_entry_price"]
                new_shares = curr_shares + shares
                new_avg = ((curr_shares * curr_avg) + (shares * current_price)) / new_shares
                self.state["positions"][ticker] = {
                    "shares": new_shares,
                    "avg_entry_price": round(new_avg, 2),
                    "stop_loss": round(stop_loss, 2) if stop_loss else self.state["positions"][ticker].get("stop_loss", 0.0),
                    "take_profit": round(take_profit, 2) if take_profit else self.state["positions"][ticker].get("take_profit", 0.0)
                }
            else:
                self.state["positions"][ticker] = {
                    "shares": shares,
                    "avg_entry_price": round(current_price, 2),
                    "stop_loss": round(stop_loss, 2) if stop_loss else 0.0,
                    "take_profit": round(take_profit, 2) if take_profit else 0.0
                }

            order_record = {
                "order_id": f"ORD-{len(self.state['order_history']) + 1:04d}",
                "timestamp": now_str,
                "ticker": ticker,
                "action": "BUY",
                "type": order_type,
                "status": "EXECUTED",
                "shares": shares,
                "price": round(current_price, 2),
                "total": round(order_cost, 2),
                "stop_loss": round(stop_loss, 2) if stop_loss else "N/A",
                "take_profit": round(take_profit, 2) if take_profit else "N/A"
            }
            self.state["order_history"].append(order_record)
            self._save_wallet()
            return {"success": True, "message": f"Successfully bought {shares} shares of {ticker} at ${round(current_price, 2)} | SL: ${stop_loss or 'None'} | TP: ${take_profit or 'None'}"}

        elif action == "SELL":
            if ticker not in self.state["positions"] or self.state["positions"][ticker]["shares"] < shares:
                available = self.state["positions"].get(ticker, {}).get("shares", 0)
                return {"success": False, "message": f"Insufficient shares to sell. Owned: {available}"}

            entry_price = self.state["positions"][ticker]["avg_entry_price"]
            realized_gain = (current_price - entry_price) * shares

            self.state["cash"] += order_cost
            self.state["realized_pnl"] += realized_gain
            self.state["positions"][ticker]["shares"] -= shares

            if self.state["positions"][ticker]["shares"] <= 0:
                del self.state["positions"][ticker]

            order_record = {
                "order_id": f"ORD-{len(self.state['order_history']) + 1:04d}",
                "timestamp": now_str,
                "ticker": ticker,
                "action": "SELL",
                "type": order_type,
                "status": "EXECUTED",
                "shares": shares,
                "price": round(current_price, 2),
                "total": round(order_cost, 2),
                "realized_pnl": round(realized_gain, 2)
            }
            self.state["order_history"].append(order_record)
            self._save_wallet()
            return {"success": True, "message": f"Successfully sold {shares} shares of {ticker} at ${round(current_price, 2)} (Realized P&L: ${round(realized_gain, 2):+,.2f})"}

        return {"success": False, "message": f"Unknown action {action}"}

    def check_and_trigger_risk_orders(self, current_prices: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Scans all open positions against live market prices to trigger automated Stop-Loss or Take-Profit exits.
        """
        triggered = []
        for ticker in list(self.state["positions"].keys()):
            pos = self.state["positions"][ticker]
            curr_price = current_prices.get(ticker)
            if not curr_price:
                continue

            sl = pos.get("stop_loss", 0.0)
            tp = pos.get("take_profit", 0.0)
            shares = pos["shares"]

            # Trigger Stop Loss
            if sl and sl > 0 and curr_price <= sl:
                res = self.execute_order(ticker=ticker, action="SELL", shares=shares, current_price=curr_price, order_type="STOP_LOSS")
                triggered.append({"ticker": ticker, "type": "STOP_LOSS", "price": curr_price, "message": res["message"]})

            # Trigger Take Profit
            elif tp and tp > 0 and curr_price >= tp:
                res = self.execute_order(ticker=ticker, action="SELL", shares=shares, current_price=curr_price, order_type="TAKE_PROFIT")
                triggered.append({"ticker": ticker, "type": "TAKE_PROFIT", "price": curr_price, "message": res["message"]})

        return triggered

    def get_portfolio_summary(self, current_prices: Dict[str, float]) -> Dict[str, Any]:
        """
        Computes real-time portfolio valuation, unrealized P&L, risk boundaries, and open positions.
        """
        holdings_value = 0.0
        positions_table = []

        for ticker, pos in self.state["positions"].items():
            shares = pos["shares"]
            avg_entry = pos["avg_entry_price"]
            curr_price = current_prices.get(ticker, avg_entry)
            market_val = shares * curr_price
            unrealized_pnl = (curr_price - avg_entry) * shares
            unrealized_pnl_pct = ((curr_price - avg_entry) / avg_entry) * 100 if avg_entry > 0 else 0.0

            holdings_value += market_val
            positions_table.append({
                "Ticker": ticker,
                "Shares": shares,
                "Avg Entry Price": f"${avg_entry:,.2f}",
                "Current Price": f"${curr_price:,.2f}",
                "Market Value": f"${market_val:,.2f}",
                "Stop Loss": f"${pos.get('stop_loss', 0.0):,.2f}" if pos.get('stop_loss') else "None",
                "Take Profit": f"${pos.get('take_profit', 0.0):,.2f}" if pos.get('take_profit') else "None",
                "Unrealized P&L": f"${unrealized_pnl:+,.2f} ({unrealized_pnl_pct:+.2f}%)",
                "Raw_PnL": unrealized_pnl,
                "Is_Profit": unrealized_pnl >= 0
            })

        total_value = self.state["cash"] + holdings_value
        total_pnl = total_value - self.initial_capital
        total_pnl_pct = (total_pnl / self.initial_capital) * 100

        return {
            "cash": round(self.state["cash"], 2),
            "holdings_value": round(holdings_value, 2),
            "total_portfolio_value": round(total_value, 2),
            "total_pnl": round(total_pnl, 2),
            "total_pnl_pct": round(total_pnl_pct, 2),
            "realized_pnl": round(self.state["realized_pnl"], 2),
            "positions_table": positions_table,
            "order_history": list(reversed(self.state.get("order_history", [])))
        }

    def reset_wallet(self):
        """Resets paper trading wallet to initial state."""
        self.state = {
            "cash": self.initial_capital,
            "portfolio_value": self.initial_capital,
            "realized_pnl": 0.0,
            "positions": {},
            "order_history": [],
            "pending_orders": []
        }
        self._save_wallet()
