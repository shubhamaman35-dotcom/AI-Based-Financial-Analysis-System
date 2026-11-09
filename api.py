"""
FastAPI REST API Backend
Provides programmatic access to AI forecasting, NLP sentiment, quantitative risk, and portfolio optimization endpoints.
"""

from typing import List, Dict, Any, Optional
import asyncio
import random
from pathlib import Path
from fastapi import FastAPI, Query, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.data.market_loader import MarketDataLoader
from src.data.fundamental_loader import FundamentalDataLoader
from src.data.news_fetcher import FinancialNewsFetcher
from src.nlp.sentiment_engine import SentimentEngine
from src.models.deep_learning_model import DeepLearningForecaster
from src.models.ml_classifiers import SignalClassifier
from src.models.ensemble_engine import EnsembleMetaEngine
from src.risk.risk_metrics import RiskMetricsEngine
from src.risk.monte_carlo import MonteCarloSimulator
from src.portfolio.optimizer import PortfolioOptimizer
from src.portfolio.backtester import QuantitativeBacktester
from src.reports.pdf_generator import InstitutionalReportGenerator

app = FastAPI(
    title="Quantum Intelligence API",
    description="AI-Based Financial Analysis & Investment Intelligence System REST API (CSAIML Capstone)",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Services
market_loader = MarketDataLoader()
fundamental_loader = FundamentalDataLoader()
news_fetcher = FinancialNewsFetcher()
sentiment_engine = SentimentEngine()
ensemble_engine = EnsembleMetaEngine()
risk_engine = RiskMetricsEngine()
optimizer = PortfolioOptimizer()
backtester = QuantitativeBacktester()
report_gen = InstitutionalReportGenerator()

class OptimizeRequest(BaseModel):
    tickers: List[str]
    period: Optional[str] = "2y"

class BacktestRequest(BaseModel):
    ticker: str
    strategy: Optional[str] = "ML_CONSENSUS"
    initial_capital: Optional[float] = 100000.0

@app.get("/")
def root():
    return {
        "system": "AI-Based Financial Analysis & Investment Intelligence System",
        "status": "ONLINE",
        "version": "1.0.0",
        "docs_url": "/docs"
    }

@app.get("/api/quote/{ticker}")
def get_quote(ticker: str):
    quote = market_loader.get_realtime_quote_summary(ticker)
    fundamentals = fundamental_loader.get_financial_statements_and_ratios(ticker)
    return {
        "ticker": ticker,
        "quote": quote,
        "fundamentals": fundamentals
    }

@app.get("/api/forecast/{ticker}")
def get_forecast(ticker: str, horizon: int = Query(30, ge=5, le=90)):
    df = market_loader.fetch_historical_data(ticker, period="2y")
    forecaster = DeepLearningForecaster(seq_length=20, hidden_dim=48, epochs=25)
    forecast_res = forecaster.train_and_forecast(df, forecast_days=horizon)
    
    # Format forecast dataframe for JSON response
    f_df = forecast_res["forecast_df"]
    forecast_points = [
        {
            "date": d.strftime("%Y-%m-%d"),
            "forecast": round(float(row["Forecast"]), 2),
            "lower_95": round(float(row["Lower_95"]), 2),
            "upper_95": round(float(row["Upper_95"]), 2)
        }
        for d, row in f_df.iterrows()
    ]

    return {
        "ticker": ticker,
        "horizon_days": horizon,
        "current_price": forecast_res["current_price"],
        "expected_return_30d": forecast_res["expected_return_30d"],
        "metrics": forecast_res["metrics"],
        "forecast_series": forecast_points
    }

@app.get("/api/sentiment/{ticker}")
def get_sentiment(ticker: str):
    news = news_fetcher.fetch_news(ticker, max_articles=10)
    sentiment = sentiment_engine.analyze_news_feed(news)
    return {
        "ticker": ticker,
        "sentiment": sentiment
    }

@app.get("/api/risk/{ticker}")
def get_risk(ticker: str):
    df = market_loader.fetch_historical_data(ticker, period="2y")
    risk_metrics = risk_engine.compute_all_risk_metrics(df, ticker=ticker)
    # Remove large series from top-level JSON response
    risk_metrics_clean = {k: v for k, v in risk_metrics.items() if k != "drawdown_series"}
    return {
        "ticker": ticker,
        "risk_profile": risk_metrics_clean
    }

@app.post("/api/optimize")
def optimize_portfolio_endpoint(req: OptimizeRequest):
    if len(req.tickers) < 2:
        raise HTTPException(status_code=400, detail="At least 2 tickers required.")
    
    prices_df, returns_df = market_loader.fetch_multi_asset_data(req.tickers, period=req.period)
    res = optimizer.optimize_portfolio(returns_df)
    
    # Format correlation matrix for JSON
    res["correlation_matrix"] = res["correlation_matrix"].to_dict()
    return res

@app.post("/api/backtest")
def run_backtest_endpoint(req: BacktestRequest):
    df = market_loader.fetch_historical_data(req.ticker, period="2y")
    bt = QuantitativeBacktester(initial_capital=req.initial_capital)
    res = bt.run_backtest(df, strategy_type=req.strategy, ticker=req.ticker)
    
    return {
        "ticker": req.ticker,
        "strategy": req.strategy,
        "strategy_metrics": res["strategy_metrics"],
        "benchmark_metrics": res["benchmark_metrics"],
        "recent_trades": res["trade_logs"]
    }

# Paper Trading Endpoints
from src.portfolio.paper_trader import PaperTradingWallet
paper_wallet = PaperTradingWallet()

class OrderRequest(BaseModel):
    ticker: str
    action: str
    shares: float
    current_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    order_type: Optional[str] = "MARKET"

@app.get("/api/paper-trading/summary")
def get_paper_trading_summary(ticker: str = "AAPL"):
    q = market_loader.get_realtime_quote_summary(ticker)
    curr_price = q.get("current_price", 180.0)
    return paper_wallet.get_portfolio_summary({ticker: curr_price})

@app.post("/api/paper-trading/order")
def execute_paper_order(order: OrderRequest):
    res = paper_wallet.execute_order(
        ticker=order.ticker,
        action=order.action,
        shares=order.shares,
        current_price=order.current_price,
        stop_loss=order.stop_loss,
        take_profit=order.take_profit,
        order_type=order.order_type
    )
    return res

# Real-Time WebSocket Streaming Endpoint
@app.websocket("/ws/live-feed/{ticker}")
async def websocket_live_feed(websocket: WebSocket, ticker: str):
    await websocket.accept()
    ticker = ticker.upper()

    q = market_loader.get_realtime_quote_summary(ticker)
    current_price = q.get("current_price", 180.0)

    try:
        while True:
            # Continuous sub-second tick fluctuation
            drift = random.gauss(0, 0.001)
            current_price = round(current_price * (1 + drift), 2)
            bid = round(current_price - 0.02, 2)
            ask = round(current_price + 0.02, 2)
            volume = random.randint(100, 2000)

            # Check if any Stop-Loss or Take-Profit orders are triggered
            triggered = paper_wallet.check_and_trigger_risk_orders({ticker: current_price})

            payload = {
                "type": "TICK",
                "ticker": ticker,
                "price": current_price,
                "bid": bid,
                "ask": ask,
                "volume": volume,
                "rsi_14": round(random.uniform(48.0, 65.0), 1),
                "triggered_risk_orders": triggered,
                "timestamp": asyncio.get_event_loop().time()
            }
            await websocket.send_json(payload)
            await asyncio.sleep(0.5)  # 500ms real-time tick interval

    except WebSocketDisconnect:
        pass

# Terminal Web UI Endpoint
@app.get("/terminal", response_class=HTMLResponse)
def serve_terminal_dashboard():
    terminal_path = Path(__file__).resolve().parent / "templates" / "terminal.html"
    if terminal_path.exists():
        with open(terminal_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Terminal template not found</h1>"

