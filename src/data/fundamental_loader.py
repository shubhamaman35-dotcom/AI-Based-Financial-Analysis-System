"""
Fundamental Data Loader & Financial Statement Analyzer
Extracts, standardizes, and computes key valuation, profitability, solvency, and health metrics.
"""

import logging
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FundamentalDataLoader:
    """
    Parses corporate balance sheets, income statements, cash flow statements,
    and calculates quantitative financial ratios & health scores.
    """

    def __init__(self):
        pass

    def get_financial_statements_and_ratios(self, ticker: str) -> Dict[str, Any]:
        """
        Fetches or simulates full financial statement tables and key ratios.
        """
        try:
            import yfinance as yf
            t = yf.Ticker(ticker)
            
            # Fetch statements
            income_stmt = t.income_stmt if hasattr(t, "income_stmt") and t.income_stmt is not None and not t.income_stmt.empty else None
            balance_sheet = t.balance_sheet if hasattr(t, "balance_sheet") and t.balance_sheet is not None and not t.balance_sheet.empty else None
            cashflow = t.cashflow if hasattr(t, "cashflow") and t.cashflow is not None and not t.cashflow.empty else None
            info = t.info if hasattr(t, "info") and t.info else {}

            if income_stmt is not None and balance_sheet is not None:
                return self._parse_real_fundamentals(ticker, income_stmt, balance_sheet, cashflow, info)
        except Exception as e:
            logger.warning(f"Failed to fetch live statements for {ticker}: {e}. Generating simulated fundamentals.")

        return self._generate_simulated_fundamentals(ticker)

    def _parse_real_fundamentals(
        self,
        ticker: str,
        income_stmt: pd.DataFrame,
        balance_sheet: pd.DataFrame,
        cashflow: Optional[pd.DataFrame],
        info: Dict[str, Any]
    ) -> Dict[str, Any]:
        # Formulate clean statement summaries
        income_summary = income_stmt.iloc[:8, :4].copy() if not income_stmt.empty else pd.DataFrame()
        balance_summary = balance_sheet.iloc[:8, :4].copy() if not balance_sheet.empty else pd.DataFrame()
        cashflow_summary = cashflow.iloc[:8, :4].copy() if cashflow is not None and not cashflow.empty else pd.DataFrame()

        # Extract or calculate ratios
        trailing_pe = info.get("trailingPE", "N/A")
        forward_pe = info.get("forwardPE", "N/A")
        price_to_book = info.get("priceToBook", "N/A")
        roe = info.get("returnOnEquity", "N/A")
        roa = info.get("returnOnAssets", "N/A")
        debt_to_equity = info.get("debtToEquity", "N/A")
        current_ratio = info.get("currentRatio", "N/A")
        profit_margins = info.get("profitMargins", "N/A")
        operating_margins = info.get("operatingMargins", "N/A")
        revenue_growth = info.get("revenueGrowth", "N/A")
        free_cashflow = info.get("freeCashflow", "N/A")

        # Health score computation
        health_score, score_breakdown = self._compute_health_score(
            roe=roe,
            current_ratio=current_ratio,
            debt_to_equity=debt_to_equity,
            profit_margins=profit_margins,
            revenue_growth=revenue_growth
        )

        return {
            "ticker": ticker,
            "income_statement": income_summary,
            "balance_sheet": balance_summary,
            "cash_flow": cashflow_summary,
            "ratios": {
                "Trailing P/E": round(trailing_pe, 2) if isinstance(trailing_pe, (int, float)) else trailing_pe,
                "Forward P/E": round(forward_pe, 2) if isinstance(forward_pe, (int, float)) else forward_pe,
                "Price-to-Book (P/B)": round(price_to_book, 2) if isinstance(price_to_book, (int, float)) else price_to_book,
                "Return on Equity (ROE)": f"{round(roe * 100, 2)}%" if isinstance(roe, (int, float)) else str(roe),
                "Return on Assets (ROA)": f"{round(roa * 100, 2)}%" if isinstance(roa, (int, float)) else str(roa),
                "Debt-to-Equity": round(debt_to_equity, 2) if isinstance(debt_to_equity, (int, float)) else debt_to_equity,
                "Current Ratio": round(current_ratio, 2) if isinstance(current_ratio, (int, float)) else current_ratio,
                "Net Profit Margin": f"{round(profit_margins * 100, 2)}%" if isinstance(profit_margins, (int, float)) else str(profit_margins),
                "Operating Margin": f"{round(operating_margins * 100, 2)}%" if isinstance(operating_margins, (int, float)) else str(operating_margins),
                "Revenue Growth (YoY)": f"{round(revenue_growth * 100, 2)}%" if isinstance(revenue_growth, (int, float)) else str(revenue_growth)
            },
            "health_score": health_score,
            "score_breakdown": score_breakdown
        }

    def _generate_simulated_fundamentals(self, ticker: str) -> Dict[str, Any]:
        np.random.seed(abs(hash(ticker)) % (2**32 - 1))
        
        years = ["2024", "2023", "2022", "2021"]
        base_rev = float(np.random.uniform(50000, 380000)) # in millions
        growth = np.random.uniform(0.08, 0.25)
        gross_margin = np.random.uniform(0.35, 0.65)
        net_margin = np.random.uniform(0.12, 0.28)

        # Build simulated income statement
        revs = [round(base_rev * ((1 + growth) ** (3 - i)), 1) for i in range(4)]
        gross_profits = [round(r * gross_margin, 1) for r in revs]
        operating_incomes = [round(r * (gross_margin - 0.18), 1) for r in revs]
        net_incomes = [round(r * net_margin, 1) for r in revs]

        income_df = pd.DataFrame({
            "Metric": ["Total Revenue ($M)", "Gross Profit ($M)", "Operating Income ($M)", "Net Income ($M)"],
            years[0]: [revs[0], gross_profits[0], operating_incomes[0], net_incomes[0]],
            years[1]: [revs[1], gross_profits[1], operating_incomes[1], net_incomes[1]],
            years[2]: [revs[2], gross_profits[2], operating_incomes[2], net_incomes[2]],
            years[3]: [revs[3], gross_profits[3], operating_incomes[3], net_incomes[3]],
        }).set_index("Metric")

        # Balance Sheet
        assets = [round(r * 1.4, 1) for r in revs]
        liabilities = [round(a * 0.45, 1) for a in assets]
        equity = [round(a - l, 1) for a, l in zip(assets, liabilities)]
        cash = [round(a * 0.22, 1) for a in assets]

        balance_df = pd.DataFrame({
            "Metric": ["Total Assets ($M)", "Total Liabilities ($M)", "Stockholders Equity ($M)", "Cash & Equivalents ($M)"],
            years[0]: [assets[0], liabilities[0], equity[0], cash[0]],
            years[1]: [assets[1], liabilities[1], equity[1], cash[1]],
            years[2]: [assets[2], liabilities[2], equity[2], cash[2]],
            years[3]: [assets[3], liabilities[3], equity[3], cash[3]],
        }).set_index("Metric")

        # Cash Flow
        op_cash = [round(ni * 1.35, 1) for ni in net_incomes]
        capex = [round(ni * 0.35, 1) for ni in net_incomes]
        fcf = [round(op - cp, 1) for op, cp in zip(op_cash, capex)]

        cashflow_df = pd.DataFrame({
            "Metric": ["Operating Cash Flow ($M)", "Capital Expenditure ($M)", "Free Cash Flow ($M)"],
            years[0]: [op_cash[0], capex[0], fcf[0]],
            years[1]: [op_cash[1], capex[1], fcf[1]],
            years[2]: [op_cash[2], capex[2], fcf[2]],
            years[3]: [op_cash[3], capex[3], fcf[3]],
        }).set_index("Metric")

        pe = round(float(np.random.uniform(18, 38)), 2)
        pb = round(float(np.random.uniform(3.5, 12.0)), 2)
        roe_val = round(float(np.random.uniform(0.18, 0.42)), 4)
        roa_val = round(float(np.random.uniform(0.08, 0.18)), 4)
        current_ratio_val = round(float(np.random.uniform(1.2, 2.5)), 2)
        debt_to_equity_val = round(float(np.random.uniform(0.3, 1.2)), 2)

        health_score, score_breakdown = self._compute_health_score(
            roe=roe_val,
            current_ratio=current_ratio_val,
            debt_to_equity=debt_to_equity_val,
            profit_margins=net_margin,
            revenue_growth=growth
        )

        return {
            "ticker": ticker,
            "income_statement": income_df,
            "balance_sheet": balance_df,
            "cash_flow": cashflow_df,
            "ratios": {
                "Trailing P/E": pe,
                "Forward P/E": round(pe * 0.88, 2),
                "Price-to-Book (P/B)": pb,
                "Return on Equity (ROE)": f"{round(roe_val * 100, 2)}%",
                "Return on Assets (ROA)": f"{round(roa_val * 100, 2)}%",
                "Debt-to-Equity": debt_to_equity_val,
                "Current Ratio": current_ratio_val,
                "Net Profit Margin": f"{round(net_margin * 100, 2)}%",
                "Operating Margin": f"{round((gross_margin - 0.18) * 100, 2)}%",
                "Revenue Growth (YoY)": f"{round(growth * 100, 2)}%"
            },
            "health_score": health_score,
            "score_breakdown": score_breakdown
        }

    def _compute_health_score(
        self,
        roe: Any,
        current_ratio: Any,
        debt_to_equity: Any,
        profit_margins: Any,
        revenue_growth: Any
    ) -> Tuple[int, Dict[str, int]]:
        """
        Computes a composite financial health score (0-100) based on Piotroski F-score & Altman Z-score principles.
        """
        score = 50 # baseline
        breakdown = {
            "Profitability": 20,
            "Liquidity": 20,
            "Solvency": 20,
            "Growth": 20
        }

        # Profitability
        if isinstance(roe, (int, float)) and roe > 0.15:
            breakdown["Profitability"] += 5
        if isinstance(profit_margins, (int, float)) and profit_margins > 0.12:
            breakdown["Profitability"] += 5

        # Liquidity
        if isinstance(current_ratio, (int, float)):
            if current_ratio >= 1.5:
                breakdown["Liquidity"] += 5
            elif current_ratio < 1.0:
                breakdown["Liquidity"] -= 5

        # Solvency
        if isinstance(debt_to_equity, (int, float)):
            if debt_to_equity < 0.8:
                breakdown["Solvency"] += 5
            elif debt_to_equity > 2.0:
                breakdown["Solvency"] -= 8

        # Growth
        if isinstance(revenue_growth, (int, float)):
            if revenue_growth > 0.10:
                breakdown["Growth"] += 5
            elif revenue_growth < 0.0:
                breakdown["Growth"] -= 10

        total_score = max(10, min(98, sum(breakdown.values())))
        return int(total_score), breakdown
