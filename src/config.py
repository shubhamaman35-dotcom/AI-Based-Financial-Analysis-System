"""
Configuration Module for AI-Based Financial Analysis & Investment Intelligence System
Defines system-wide constants, default asset universes, model hyperparameters, and risk settings.
"""

from pathlib import Path
from typing import Dict, List, Any

# Base Directory Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_CACHE_DIR = BASE_DIR / "cache"
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR = BASE_DIR / "models_saved"

# Create directories if they don't exist
for folder in [DATA_CACHE_DIR, REPORTS_DIR, MODELS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# Default Asset Universes
DEFAULT_ASSET_UNIVERSES: Dict[str, List[str]] = {
    "US Mega Cap Tech": ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA"],
    "US Large Cap Indices": ["SPY", "QQQ", "DIA", "IWM"],
    "Global & Sector ETFs": ["XLK", "XLF", "XLE", "XLV", "SMH", "GLD", "SLV"],
    "Indian Bluechips (NSE)": ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "TATAMOTORS.NS"],
    "Cryptocurrency": ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "ADA-USD"],
    "Forex & Commodities": ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "GC=F", "CL=F"]
}

DEFAULT_BENCHMARK = "SPY"
DEFAULT_TICKER = "AAPL"
DEFAULT_PERIOD = "2y"
DEFAULT_INTERVAL = "1d"

# Technical Indicators Parameters
INDICATOR_PARAMS: Dict[str, Any] = {
    "rsi_period": 14,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "bb_period": 20,
    "bb_std": 2.0,
    "sma_short": 20,
    "sma_medium": 50,
    "sma_long": 200,
    "ema_short": 12,
    "ema_long": 26,
    "atr_period": 14,
    "stoch_k": 14,
    "stoch_d": 3,
    "adx_period": 14,
    "williams_period": 14
}

# Machine Learning & Deep Learning Hyperparameters
ML_CONFIG: Dict[str, Any] = {
    "sequence_length": 30,       # Lookback window for LSTM/GRU
    "forecast_horizons": [1, 5, 10, 20, 30], # Forecast days ahead
    "test_split_ratio": 0.20,
    "random_state": 42,
    "lstm_hidden_dim": 64,
    "lstm_layers": 2,
    "dropout_rate": 0.2,
    "learning_rate": 0.005,
    "epochs": 40,
    "batch_size": 32,
    "n_estimators_rf": 100,
    "n_estimators_gb": 150
}

# Risk Management Parameters
RISK_CONFIG: Dict[str, Any] = {
    "risk_free_rate": 0.045,      # 4.5% annual risk-free rate
    "confidence_level_var": 0.95,  # 95% VaR
    "confidence_level_cvar": 0.99, # 99% CVaR
    "monte_carlo_simulations": 1000,
    "monte_carlo_days": 30,
    "stress_scenarios": {
        "2008 Financial Crisis": -0.40,
        "2020 COVID Shock": -0.34,
        "2022 Tech Selloff": -0.28,
        "Flash Crash Shock": -0.15
    }
}

# UI & Color Theme
THEME_CONFIG: Dict[str, str] = {
    "background_color": "#0E1117",
    "card_background": "#1E222D",
    "primary_accent": "#00FFA3",      # Cyber Cyan / Green
    "secondary_accent": "#00B4D8",    # Electric Blue
    "danger_color": "#FF4B4B",        # Bearish Red
    "warning_color": "#FFAA00",       # Neutral Amber
    "text_primary": "#FFFFFF",
    "text_secondary": "#9CA3AF"
}
