"""
Modern Portfolio Theory (MPT) Optimizer Module
Implements Markowitz Mean-Variance Optimization, Efficient Frontier calculation,
Max Sharpe, Min Volatility, and Inverse Volatility Risk Parity allocations.
"""

import logging
from typing import List, Dict, Tuple, Any
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.config import RISK_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PortfolioOptimizer:
    """
    Optimizes multi-asset portfolios across the Markowitz Efficient Frontier.
    """

    def __init__(self, risk_free_rate: float = RISK_CONFIG["risk_free_rate"]):
        self.risk_free_rate = risk_free_rate

    def optimize_portfolio(
        self,
        returns_df: pd.DataFrame,
        num_frontier_points: int = 50
    ) -> Dict[str, Any]:
        """
        Runs full Modern Portfolio Theory optimization over historical returns.
        """
        tickers = list(returns_df.columns)
        n_assets = len(tickers)

        if n_assets < 2:
            raise ValueError("Portfolio optimization requires at least 2 distinct assets.")

        # Annualized Mean Returns and Covariance Matrix
        mean_returns = returns_df.mean() * 252
        cov_matrix = returns_df.cov() * 252
        corr_matrix = returns_df.corr()

        # 1. Max Sharpe Ratio Portfolio Optimization
        max_sharpe_weights, max_sharpe_ret, max_sharpe_vol, max_sharpe_val = self._optimize_max_sharpe(
            mean_returns, cov_matrix, n_assets
        )

        # 2. Min Volatility Portfolio Optimization
        min_vol_weights, min_vol_ret, min_vol_vol, min_vol_sharpe = self._optimize_min_volatility(
            mean_returns, cov_matrix, n_assets
        )

        # 3. Equal Weight Benchmark (1/N)
        eq_weights = np.ones(n_assets) / n_assets
        eq_ret = float(np.sum(mean_returns * eq_weights))
        eq_vol = float(np.sqrt(np.dot(eq_weights.T, np.dot(cov_matrix, eq_weights))))
        eq_sharpe = float((eq_ret - self.risk_free_rate) / (eq_vol + 1e-8))

        # 4. Inverse Volatility Risk Parity Weighting
        asset_vols = returns_df.std() * np.sqrt(252)
        inv_vols = 1.0 / (asset_vols + 1e-8)
        inv_weights = (inv_vols / inv_vols.sum()).values
        inv_ret = float(np.sum(mean_returns * inv_weights))
        inv_vol = float(np.sqrt(np.dot(inv_weights.T, np.dot(cov_matrix, inv_weights))))
        inv_sharpe = float((inv_ret - self.risk_free_rate) / (inv_vol + 1e-8))

        # 5. Generate Efficient Frontier Curve Points
        frontier_vols, frontier_rets = self._calculate_efficient_frontier(
            mean_returns, cov_matrix, n_assets, min_vol_ret, max(mean_returns.max(), max_sharpe_ret * 1.2), num_frontier_points
        )

        # Format allocation weight dictionaries
        max_sharpe_alloc = {t: round(float(w) * 100, 2) for t, w in zip(tickers, max_sharpe_weights)}
        min_vol_alloc = {t: round(float(w) * 100, 2) for t, w in zip(tickers, min_vol_weights)}
        equal_weight_alloc = {t: round(float(w) * 100, 2) for t, w in zip(tickers, eq_weights)}
        risk_parity_alloc = {t: round(float(w) * 100, 2) for t, w in zip(tickers, inv_weights)}

        # Random Portfolios Monte Carlo Scatter (for frontier cloud visualization)
        mc_vols, mc_rets, mc_sharpes = self._generate_random_portfolios(mean_returns, cov_matrix, n_assets, 1000)

        return {
            "tickers": tickers,
            "correlation_matrix": corr_matrix.round(3),
            "max_sharpe_portfolio": {
                "weights": max_sharpe_alloc,
                "expected_return": round(max_sharpe_ret * 100, 2),
                "volatility": round(max_sharpe_vol * 100, 2),
                "sharpe_ratio": round(max_sharpe_val, 2)
            },
            "min_volatility_portfolio": {
                "weights": min_vol_alloc,
                "expected_return": round(min_vol_ret * 100, 2),
                "volatility": round(min_vol_vol * 100, 2),
                "sharpe_ratio": round(min_vol_sharpe, 2)
            },
            "equal_weight_portfolio": {
                "weights": equal_weight_alloc,
                "expected_return": round(eq_ret * 100, 2),
                "volatility": round(eq_vol * 100, 2),
                "sharpe_ratio": round(eq_sharpe, 2)
            },
            "risk_parity_portfolio": {
                "weights": risk_parity_alloc,
                "expected_return": round(inv_ret * 100, 2),
                "volatility": round(inv_vol * 100, 2),
                "sharpe_ratio": round(inv_sharpe, 2)
            },
            "efficient_frontier": {
                "volatilities": [round(v * 100, 2) for v in frontier_vols],
                "returns": [round(r * 100, 2) for r in frontier_rets]
            },
            "monte_carlo_cloud": {
                "volatilities": [round(v * 100, 2) for v in mc_vols],
                "returns": [round(r * 100, 2) for r in mc_rets],
                "sharpes": [round(s, 2) for s in mc_sharpes]
            }
        }

    def _optimize_max_sharpe(self, mean_returns, cov_matrix, n_assets):
        def neg_sharpe(weights):
            p_ret = np.sum(mean_returns * weights)
            p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return -(p_ret - self.risk_free_rate) / (p_vol + 1e-8)

        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
        bounds = tuple((0.0, 1.0) for _ in range(n_assets))
        init_weights = np.ones(n_assets) / n_assets

        res = minimize(neg_sharpe, init_weights, method="SLSQP", bounds=bounds, constraints=constraints)
        opt_weights = res.x
        opt_ret = float(np.sum(mean_returns * opt_weights))
        opt_vol = float(np.sqrt(np.dot(opt_weights.T, np.dot(cov_matrix, opt_weights))))
        opt_sharpe = float((opt_ret - self.risk_free_rate) / (opt_vol + 1e-8))

        return opt_weights, opt_ret, opt_vol, opt_sharpe

    def _optimize_min_volatility(self, mean_returns, cov_matrix, n_assets):
        def portfolio_vol(weights):
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
        bounds = tuple((0.0, 1.0) for _ in range(n_assets))
        init_weights = np.ones(n_assets) / n_assets

        res = minimize(portfolio_vol, init_weights, method="SLSQP", bounds=bounds, constraints=constraints)
        opt_weights = res.x
        opt_ret = float(np.sum(mean_returns * opt_weights))
        opt_vol = float(np.sqrt(np.dot(opt_weights.T, np.dot(cov_matrix, opt_weights))))
        opt_sharpe = float((opt_ret - self.risk_free_rate) / (opt_vol + 1e-8))

        return opt_weights, opt_ret, opt_vol, opt_sharpe

    def _calculate_efficient_frontier(self, mean_returns, cov_matrix, n_assets, min_ret, max_ret, num_points):
        target_returns = np.linspace(min_ret, max_ret, num_points)
        frontier_vols = []
        frontier_rets = []

        bounds = tuple((0.0, 1.0) for _ in range(n_assets))
        init_weights = np.ones(n_assets) / n_assets

        for t_ret in target_returns:
            constraints = (
                {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
                {"type": "eq", "fun": lambda w: np.sum(mean_returns * w) - t_ret}
            )
            res = minimize(
                lambda w: np.sqrt(np.dot(w.T, np.dot(cov_matrix, w))),
                init_weights,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints
            )
            if res.success:
                frontier_vols.append(float(res.fun))
                frontier_rets.append(float(t_ret))

        return frontier_vols, frontier_rets

    def _generate_random_portfolios(self, mean_returns, cov_matrix, n_assets, n_portfolios=1000):
        np.random.seed(42)
        weights_matrix = np.random.dirichlet(np.ones(n_assets), size=n_portfolios)
        returns = np.dot(weights_matrix, mean_returns)
        vols = np.array([np.sqrt(np.dot(w.T, np.dot(cov_matrix, w))) for w in weights_matrix])
        sharpes = (returns - self.risk_free_rate) / (vols + 1e-8)
        return vols, returns, sharpes
