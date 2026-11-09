"""
Risk Metrics and Stress Testing Module
Calculates VaR, CVaR, Drawdown, Sharpe/Sortino ratios, and executes historical stress scenario shocks.
"""

import logging
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from scipy import stats

from src.config import RISK_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RiskMetricsEngine:
    """
    Computes parametric and non-parametric Value-at-Risk, Expected Shortfall,
    downside volatility, maximum drawdown profiles, and stress test scenarios.
    """

    def __init__(self, risk_free_rate: float = RISK_CONFIG["risk_free_rate"]):
        self.risk_free_rate = risk_free_rate

    def compute_all_risk_metrics(self, df: pd.DataFrame, ticker: str = "ASSET") -> Dict[str, Any]:
        """
        Computes comprehensive institutional risk dashboard metrics for a price series.
        """
        returns = df["Close"].pct_change().dropna().values
        if len(returns) < 10:
            raise ValueError("Insufficient return data for risk computation.")

        # Annualization factors
        trading_days = 252
        ann_mean_return = float(np.mean(returns) * trading_days)
        ann_volatility = float(np.std(returns) * np.sqrt(trading_days))

        # 1. Value at Risk (VaR 95% & 99% - 1 Day & 30 Day)
        hist_var_95_1d = float(-np.percentile(returns, 5) * 100)
        hist_var_99_1d = float(-np.percentile(returns, 1) * 100)
        param_var_95_1d = float(-(np.mean(returns) - 1.645 * np.std(returns)) * 100)
        param_var_99_1d = float(-(np.mean(returns) - 2.326 * np.std(returns)) * 100)

        # 30-day VaR (scaled by sqrt(30))
        var_95_30d = round(hist_var_95_1d * np.sqrt(30), 2)
        var_99_30d = round(hist_var_99_1d * np.sqrt(30), 2)

        # 2. Conditional Value at Risk (CVaR / Expected Shortfall)
        tail_5 = returns[returns <= np.percentile(returns, 5)]
        cvar_95_1d = float(-np.mean(tail_5) * 100) if len(tail_5) > 0 else hist_var_95_1d
        tail_1 = returns[returns <= np.percentile(returns, 1)]
        cvar_99_1d = float(-np.mean(tail_1) * 100) if len(tail_1) > 0 else hist_var_99_1d
        cvar_95_30d = round(cvar_95_1d * np.sqrt(30), 2)

        # 3. Maximum Drawdown & Drawdown Series
        cum_returns = (1 + df["Close"].pct_change().fillna(0)).cumprod()
        running_max = cum_returns.cummax()
        drawdown_series = (cum_returns - running_max) / running_max
        max_drawdown = float(drawdown_series.min() * 100)

        # Drawdown duration in days
        is_drawdown = drawdown_series < 0
        dd_periods = is_drawdown.groupby((~is_drawdown).cumsum()).sum()
        max_dd_duration_days = int(dd_periods.max()) if not dd_periods.empty else 0

        # 4. Risk-Adjusted Return Ratios
        excess_return = ann_mean_return - self.risk_free_rate
        sharpe_ratio = float(excess_return / (ann_volatility + 1e-8))

        # Downside Deviation for Sortino Ratio
        downside_returns = returns[returns < 0]
        downside_std = float(np.std(downside_returns) * np.sqrt(trading_days)) if len(downside_returns) > 0 else ann_volatility
        sortino_ratio = float(excess_return / (downside_std + 1e-8))

        # Calmar Ratio (Annualized Return / Max Drawdown)
        calmar_ratio = float(ann_mean_return / (abs(max_drawdown / 100.0) + 1e-8))

        # 5. Skewness and Kurtosis
        skew = float(stats.skew(returns))
        kurt = float(stats.kurtosis(returns))

        # 6. Stress Testing Scenarios
        stress_results = self.execute_stress_test(current_price=float(df["Close"].iloc[-1]))

        return {
            "ticker": ticker,
            "annualized_return": round(ann_mean_return * 100, 2),
            "annualized_volatility": round(ann_volatility * 100, 2),
            "sharpe_ratio": round(sharpe_ratio, 2),
            "sortino_ratio": round(sortino_ratio, 2),
            "calmar_ratio": round(calmar_ratio, 2),
            "max_drawdown": round(max_drawdown, 2),
            "max_drawdown_duration_days": max_dd_duration_days,
            "var_95_1d": round(hist_var_95_1d, 2),
            "var_99_1d": round(hist_var_99_1d, 2),
            "var_95_30d": var_95_30d,
            "var_99_30d": var_99_30d,
            "cvar_95_1d": round(cvar_95_1d, 2),
            "cvar_95_30d": cvar_95_30d,
            "skewness": round(skew, 2),
            "kurtosis": round(kurt, 2),
            "drawdown_series": drawdown_series,
            "stress_tests": stress_results
        }

    def execute_stress_test(self, current_price: float) -> List[Dict[str, Any]]:
        """
        Applies historical macroeconomic crisis shocks to estimate potential portfolio drawdown and capital at risk.
        """
        scenarios = RISK_CONFIG["stress_scenarios"]
        results = []

        for scenario_name, shock_pct in scenarios.items():
            estimated_price = current_price * (1 + shock_pct)
            loss_per_share = current_price - estimated_price
            results.append({
                "scenario": scenario_name,
                "shock_percentage": f"{int(shock_pct * 100)}%",
                "projected_price": round(estimated_price, 2),
                "loss_per_share": round(loss_per_share, 2),
                "severity": "Extreme" if abs(shock_pct) >= 0.30 else "Moderate"
            })

        return results
