"""
Modular UI Views Package
"""
from .view_market import render_market_view
from .view_forecast import render_forecast_view
from .view_sentiment import render_sentiment_view
from .view_risk import render_risk_view
from .view_portfolio import render_portfolio_view
from .view_backtest import render_backtest_view
from .view_copilot import render_copilot_view
from .view_paper_trading import render_paper_trading_view

__all__ = [
    "render_market_view",
    "render_forecast_view",
    "render_sentiment_view",
    "render_risk_view",
    "render_portfolio_view",
    "render_backtest_view",
    "render_copilot_view",
    "render_paper_trading_view"
]
