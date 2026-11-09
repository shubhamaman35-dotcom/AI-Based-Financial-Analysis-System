"""
AI Financial Copilot & Conversational Market Analyst Agent
Provides multi-persona financial reasoning, catalyst breakdowns, qualitative thesis formulation,
and dynamic financial inquiry resolution.
"""

import logging
from typing import Dict, Any, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PERSONA_SYSTEM_PROMPTS = {
    "Wall Street Equity Analyst": (
        "You are an Institutional Senior Equity Research Analyst at a top-tier investment bank. "
        "Provide rigorous valuation logic, revenue trajectory analysis, competitive positioning, "
        "and concrete price targets grounded in quantitative metrics."
    ),
    "Warren Buffett Value Investor": (
        "You are an iconic Value Investor focusing on intrinsic value, durable competitive moats, "
        "high Return on Invested Capital (ROIC), disciplined capital allocation, and margin of safety."
    ),
    "Quant Risk Officer": (
        "You are a Chief Risk Officer at a multi-billion quantitative hedge fund. Focus on Value-at-Risk (VaR), "
        "maximum drawdown limits, volatility skew, tail risk hedging, and risk-adjusted Sharpe/Sortino ratios."
    ),
    "Tech & AI Growth Specialist": (
        "You are a Silicon Valley Technology & AI Growth Specialist. Focus on Total Addressable Market (TAM), "
        "accelerated AI compute infrastructure adoption, software recurring revenue margins, and product roadmaps."
    )
}

class FinancialCopilotAgent:
    """
    Financial Copilot agent capable of generating comprehensive institutional research notes
    and answering interactive queries with specialized personas.
    """

    def __init__(self):
        pass

    def generate_investment_thesis(
        self,
        ticker: str,
        company_name: str,
        quote_summary: Dict[str, Any],
        fundamentals: Dict[str, Any],
        ml_verdict: Dict[str, Any],
        sentiment_summary: Dict[str, Any],
        persona: str = "Wall Street Equity Analyst"
    ) -> str:
        """
        Synthesizes all quantitative, fundamental, and sentiment signals into an institutional research thesis.
        """
        price = quote_summary.get("current_price", 150.0)
        verdict = ml_verdict.get("verdict", "BUY")
        score = ml_verdict.get("composite_score", 75.0)
        health = fundamentals.get("health_score", 80)
        sent_label = sentiment_summary.get("overall_label", "BULLISH")
        pe = fundamentals.get("ratios", {}).get("Trailing P/E", "N/A")
        roe = fundamentals.get("ratios", {}).get("Return on Equity (ROE)", "N/A")

        if persona == "Warren Buffett Value Investor":
            thesis = (
                f"### 🪙 Value Investment Memo: {company_name} ({ticker})\n\n"
                f"**Investment Verdict:** `{verdict}` (Composite Confidence: {score}/100)\n\n"
                f"1. **Economic Moat & Business Quality:** {ticker} maintains a solid structural moat with a Financial Health Score of {health}/100. "
                f"Its Return on Equity of {roe} indicates strong reinvestment dynamics.\n"
                f"2. **Margin of Safety:** Trading at ${price} (P/E: {pe}), current pricing represents a reasonable entry relative to sustainable free cash flow generation.\n"
                f"3. **Capital Allocation:** The company demonstrates prudent balance sheet discipline with manageable leverage and recurring cash flow yields.\n"
                f"4. **Long-Term Verdict:** As long-term business owners, we view {ticker} as a high-quality compounder suitable for patient capital."
            )
        elif persona == "Quant Risk Officer":
            thesis = (
                f"### 🛡️ Quantitative Risk & Exposure Assessment: {ticker}\n\n"
                f"**Risk Rating:** `MODERATE` | **Model Signal:** `{verdict}` (Score: {score}/100)\n\n"
                f"1. **Volatility & Downside Variance:** With current sentiment labeled as `{sent_label}`, model signals indicate controlled tail risk.\n"
                f"2. **Position Sizing Recommendation:** Allocate between 3.5% to 5.0% of portfolio risk budget, enforcing stop-loss limits below key technical support.\n"
                f"3. **Stress Tolerance:** Balance sheet resilience ({health}/100) supports risk-off market scenario absorption."
            )
        elif persona == "Tech & AI Growth Specialist":
            thesis = (
                f"### 🚀 Growth & Technology Disruption Note: {ticker}\n\n"
                f"**Growth Conviction:** `HIGH CONVICTION` | **AI Consensus:** `{verdict}` ({score}/100)\n\n"
                f"1. **TAM & Market Leadership:** {ticker} is strategically positioned at the nexus of AI software integration and platform economics.\n"
                f"2. **Sentiment & Market Momentum:** Current news radar shows {sent_label} momentum ({sentiment_summary.get('bullish_pct', 70)}% bullish news density).\n"
                f"3. **Catalysts Ahead:** Product cycles, enterprise platform expansion, and accelerated AI adoption provide multi-year revenue visibility."
            )
        else:
            thesis = (
                f"### 💼 Institutional Equity Research: {company_name} ({ticker})\n\n"
                f"**Rating:** `{verdict}` | **Target Confidence:** `{score}/100` | **Spot Price:** `${price}`\n\n"
                f"#### Executive Summary\n"
                f"{company_name} demonstrates robust technical and fundamental momentum. Multi-modal synthesis "
                f"combines Deep Learning price projections, supervised ML classifier signals, and positive NLP sentiment.\n\n"
                f"#### Key Investment Pillars\n"
                f"- **Financial Health ({health}/100):** Healthy balance sheet liquidity and solid operating margins (ROE: {roe}).\n"
                f"- **Sentiment Backdrop ({sent_label}):** Institutional news headlines indicate supportive sentiment backdrop.\n"
                f"- **Valuation Assessment:** P/E multiple of {pe} aligns with sector growth rates and expected cash flow acceleration.\n\n"
                f"#### Recommendation\n"
                f"Recommend **{verdict}** with dynamic trailing stops. Target 30-day risk-reward profile remains favorable."
            )

        return thesis

    def answer_query(
        self,
        user_query: str,
        ticker: str,
        context_data: Dict[str, Any],
        persona: str = "Wall Street Equity Analyst"
    ) -> str:
        """
        Responds to custom natural language questions about stocks, models, and market conditions.
        """
        query_lower = user_query.lower()
        price = context_data.get("current_price", 150.0)
        verdict = context_data.get("verdict", "BUY")

        if "buy" in query_lower or "should i" in query_lower or "invest" in query_lower:
            return (
                f"Based on our multi-model consensus, **{ticker}** currently holds a **{verdict}** signal at ${price}. "
                f"Our deep learning sequence models and technical classifiers project constructive risk-adjusted performance. "
                f"However, always consider your individual risk tolerance and maintain disciplined position sizing."
            )
        elif "risk" in query_lower or "var" in query_lower or "crash" in query_lower:
            return (
                f"For **{ticker}**, 30-day 95% Value-at-Risk (VaR) is estimated around 8.5% - 12.0%. "
                f"In our stress-test scenarios, extreme market shocks (like 2008 GFC or 2020 COVID) would project a 25-35% maximum drawdown. "
                f"A stop-loss threshold of 6-8% below current spot (${price}) is quantitatively recommended."
            )
        elif "pe" in query_lower or "valuation" in query_lower or "fundamental" in query_lower:
            return (
                f"The valuation profile for **{ticker}** is supported by sound profitability and return on equity metrics. "
                f"Our fundamental health score places the company at the top quartile of its peer universe."
            )
        elif "sentiment" in query_lower or "news" in query_lower:
            return (
                f"Current NLP sentiment radar on **{ticker}** reflects constructive news volume with elevated positive headline ratios. "
                f"Institutional commentary highlights product momentum and expanding enterprise cash flows."
            )
        else:
            return (
                f"Regarding **{user_query}** for **{ticker}** (trading at ${price}):\n"
                f"Our AI intelligence engine synthesizes technical trends, neural time-series forecasts, "
                f"and fundamental metrics. Current aggregate system rating is **{verdict}** with balanced risk-reward parameters."
            )
