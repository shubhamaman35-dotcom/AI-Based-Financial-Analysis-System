"""
Monte Carlo Simulation Engine
Simulates multi-path asset price trajectories using Geometric Brownian Motion (GBM)
and Merton Jump Diffusion stochastic processes.
"""

import logging
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from scipy import stats

from src.config import RISK_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MonteCarloSimulator:
    """
    Performs stochastic Monte Carlo price projections and probability distribution estimations.
    """

    def __init__(self, num_simulations: int = 1000, days: int = 30):
        self.num_simulations = num_simulations
        self.days = days

    def run_simulation(
        self,
        df: pd.DataFrame,
        days: int = 30,
        num_simulations: int = 1000,
        model_type: str = "GBM_JUMP" # "GBM" or "GBM_JUMP"
    ) -> Dict[str, Any]:
        """
        Executes Monte Carlo simulation given historical price series.
        """
        prices = df["Close"].values
        current_price = float(prices[-1])
        returns = df["Close"].pct_change().dropna().values
        
        # Calculate daily drift (mu) and daily volatility (sigma)
        mu = np.mean(returns)
        sigma = np.std(returns)

        dt = 1.0 # 1 day step
        np.random.seed(42)

        # Preallocate simulation matrix: shape = (days + 1, num_simulations)
        simulation_paths = np.zeros((days + 1, num_simulations))
        simulation_paths[0] = current_price

        for t in range(1, days + 1):
            # Standard Gaussian random shocks
            z = np.random.standard_normal(num_simulations)
            
            if model_type == "GBM_JUMP":
                # Merton Jump Diffusion: Poisson process with jump size distribution
                lambda_jump = 0.05  # Jump frequency
                jump_mu = -0.01     # Average jump size
                jump_sigma = 0.04   # Jump volatility
                
                # Number of jumps per path in interval dt
                n_jumps = np.random.poisson(lambda_jump, num_simulations)
                jump_multiplier = np.exp(n_jumps * jump_mu + np.sqrt(n_jumps) * jump_sigma * np.random.randn(num_simulations))
                
                # Geometric Brownian Motion step + Jump
                drift = (mu - 0.5 * (sigma ** 2)) * dt
                diffusion = sigma * np.sqrt(dt) * z
                simulation_paths[t] = simulation_paths[t - 1] * np.exp(drift + diffusion) * jump_multiplier
            else:
                # Pure Geometric Brownian Motion
                drift = (mu - 0.5 * (sigma ** 2)) * dt
                diffusion = sigma * np.sqrt(dt) * z
                simulation_paths[t] = simulation_paths[t - 1] * np.exp(drift + diffusion)

        # Future simulation date index
        last_date = df.index[-1]
        sim_dates = [last_date + pd.Timedelta(days=i) for i in range(days + 1)]
        
        paths_df = pd.DataFrame(simulation_paths, index=sim_dates)

        # Terminal statistics (Day N values)
        terminal_prices = simulation_paths[-1, :]
        terminal_returns = (terminal_prices - current_price) / current_price

        # Quantile Trajectories
        percentiles = [5, 25, 50, 75, 95]
        quantile_trajectories = {
            f"P_{p}": np.percentile(simulation_paths, p, axis=1) for p in percentiles
        }
        quantile_df = pd.DataFrame(quantile_trajectories, index=sim_dates)

        # Key Statistical Distributions
        expected_terminal_price = float(np.mean(terminal_prices))
        median_terminal_price = float(np.median(terminal_prices))
        prob_profit = float(np.mean(terminal_returns > 0) * 100)
        prob_loss_10pct = float(np.mean(terminal_returns < -0.10) * 100)
        prob_gain_10pct = float(np.mean(terminal_returns > 0.10) * 100)

        # Monte Carlo VaR (Value at Risk) from simulated terminal distribution
        mc_var_95 = float(-np.percentile(terminal_returns, 5) * 100)
        mc_var_99 = float(-np.percentile(terminal_returns, 1) * 100)
        
        # Monte Carlo CVaR (Expected Shortfall)
        tail_5 = terminal_returns[terminal_returns <= np.percentile(terminal_returns, 5)]
        mc_cvar_95 = float(-np.mean(tail_5) * 100) if len(tail_5) > 0 else mc_var_95

        return {
            "days": days,
            "num_simulations": num_simulations,
            "current_price": round(current_price, 2),
            "expected_price": round(expected_terminal_price, 2),
            "median_price": round(median_terminal_price, 2),
            "p5_price": round(float(np.percentile(terminal_prices, 5)), 2),
            "p95_price": round(float(np.percentile(terminal_prices, 95)), 2),
            "prob_profit": round(prob_profit, 2),
            "prob_loss_10pct": round(prob_loss_10pct, 2),
            "prob_gain_10pct": round(prob_gain_10pct, 2),
            "mc_var_95": round(mc_var_95, 2),
            "mc_var_99": round(mc_var_99, 2),
            "mc_cvar_95": round(mc_cvar_95, 2),
            "paths_sample": paths_df.iloc[:, :50], # Sample 50 paths for UI rendering
            "quantile_df": quantile_df,
            "terminal_prices": terminal_prices
        }
