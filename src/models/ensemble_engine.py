"""
Ensemble Meta-Learner & Consensus Signal Synthesis Module
Combines Deep Learning Sequence predictions, Tree Classifier probabilities,
NLP Sentiment scores, and Financial Health indicators into a unified Investment Verdict.
"""

import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnsembleMetaEngine:
    """
    Blends multi-modal predictive components into an institutional Investment Intelligence Score.
    """

    def __init__(self):
        # Component Weights
        self.w_ml_class = 0.35
        self.w_dl_forecast = 0.30
        self.w_sentiment = 0.20
        self.w_health = 0.15

    def synthesize_investment_verdict(
        self,
        ml_bullish_prob: float,       # 0 - 100
        dl_expected_return_30d: float, # Percentage return, e.g. +5.4%
        sentiment_score: float,       # -1.0 to +1.0
        health_score: int             # 0 - 100
    ) -> Dict[str, Any]:
        """
        Synthesizes individual signals into a composite score (0-100) and actionable verdict.
        """
        # Normalize DL Expected Return to 0-100 scale (Assuming -15% = 0, 0% = 50, +15% = 100)
        dl_score = min(100.0, max(0.0, 50.0 + (dl_expected_return_30d * 3.33)))

        # Normalize Sentiment (-1 to +1 -> 0 to 100)
        sent_score_norm = (sentiment_score + 1.0) * 50.0

        # Health score is already 0-100
        health_score_norm = float(health_score)

        # Composite Weighted Score
        composite_score = (
            (self.w_ml_class * ml_bullish_prob) +
            (self.w_dl_forecast * dl_score) +
            (self.w_sentiment * sent_score_norm) +
            (self.w_health * health_score_norm)
        )
        composite_score = round(min(99.0, max(1.0, composite_score)), 1)

        # Final Rating Determination
        if composite_score >= 75.0:
            verdict = "STRONG BUY"
            action_code = "ACCUMULATE"
            verdict_color = "#00FFA3"
            summary_desc = "Highly favorable convergence across predictive deep learning, ML momentum signals, bullish sentiment, and balance sheet fundamentals."
        elif composite_score >= 60.0:
            verdict = "BUY"
            action_code = "OVERWEIGHT"
            verdict_color = "#48CAE4"
            summary_desc = "Positive risk-reward profile with constructive technical trend and supportive sentiment indicators."
        elif composite_score >= 45.0:
            verdict = "HOLD / NEUTRAL"
            action_code = "NEUTRAL"
            verdict_color = "#FFAA00"
            summary_desc = "Balanced risk-reward profile. Recommend maintaining existing allocation while monitoring breakout or pullback levels."
        elif composite_score >= 30.0:
            verdict = "REDUCE / SELL"
            action_code = "UNDERWEIGHT"
            verdict_color = "#FF7B00"
            summary_desc = "Elevated downside risk, decelerating momentum, or negative sentiment divergence suggests trimming exposure."
        else:
            verdict = "STRONG SELL"
            action_code = "EXIT / HEDGE"
            verdict_color = "#FF4B4B"
            summary_desc = "High probability of further downside across neural forecasts, tree classifiers, and negative sentiment profile."

        return {
            "composite_score": composite_score,
            "verdict": verdict,
            "action_code": action_code,
            "verdict_color": verdict_color,
            "summary_description": summary_desc,
            "component_breakdown": {
                "ML Classification Probability": f"{round(ml_bullish_prob, 1)}%",
                "Deep Learning 30d Expected Return": f"{round(dl_expected_return_30d, 2)}%",
                "NLP Sentiment Score": round(sentiment_score, 2),
                "Fundamental Health Score": f"{health_score}/100"
            },
            "weight_distribution": {
                "ML Classifiers": f"{int(self.w_ml_class * 100)}%",
                "Deep Learning Sequence": f"{int(self.w_dl_forecast * 100)}%",
                "NLP Sentiment Radar": f"{int(self.w_sentiment * 100)}%",
                "Fundamental Health": f"{int(self.w_health * 100)}%"
            }
        }
