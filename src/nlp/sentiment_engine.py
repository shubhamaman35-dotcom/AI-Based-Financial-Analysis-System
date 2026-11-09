"""
Financial Sentiment Analysis Engine
Evaluates market sentiment using financial domain lexicons, VADER, and polarity modeling.
"""

import logging
import re
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Financial-specific sentiment lexicon weights (Loughran-McDonald inspired)
FINANCIAL_LEXICON: Dict[str, float] = {
    "surge": 2.5, "surges": 2.5, "surging": 2.5, "rally": 2.2, "rallies": 2.2,
    "soar": 2.8, "soars": 2.8, "bullish": 2.6, "outperform": 2.4, "beat": 2.0,
    "beats": 2.0, "breakout": 2.3, "record": 1.8, "growth": 1.5, "profit": 1.6,
    "upgrade": 2.2, "upgrades": 2.2, "dividend": 1.2, "buyback": 1.7, "inflow": 1.6,
    "plummet": -2.8, "plummets": -2.8, "crash": -3.2, "bearish": -2.5, "underperform": -2.4,
    "miss": -2.1, "misses": -2.1, "drop": -1.8, "drops": -1.8, "loss": -1.9,
    "downgrade": -2.4, "downgrades": -2.4, "recession": -2.6, "inflation": -1.4,
    "scrutiny": -1.8, "lawsuit": -2.2, "subpoena": -2.5, "bankrupt": -3.5, "default": -3.0
}

class SentimentEngine:
    """
    NLP Sentiment Analysis engine optimized for financial headlines,
    earnings summaries, and analyst reports.
    """

    def __init__(self):
        self._vader_analyzer = None
        self._init_vader()

    def _init_vader(self):
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self._vader_analyzer = SentimentIntensityAnalyzer()
            # Update VADER lexicon with financial domain words
            self._vader_analyzer.lexicon.update(FINANCIAL_LEXICON)
        except Exception as e:
            logger.info(f"VADER Sentiment library loading: {e}. Using native financial rule-based parser.")

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyzes a single headline/text and returns sentiment scores.
        """
        if not text:
            return {"compound": 0.0, "label": "NEUTRAL", "pos": 0.33, "neu": 0.34, "neg": 0.33}

        if self._vader_analyzer is not None:
            scores = self._vader_analyzer.polarity_scores(text)
            compound = scores["compound"]
            pos = scores["pos"]
            neu = scores["neu"]
            neg = scores["neg"]
        else:
            # Native fallback rule-based analyzer
            compound, pos, neu, neg = self._rule_based_sentiment(text)

        # Classify label
        if compound >= 0.15:
            label = "BULLISH"
            color = "#00FFA3"
        elif compound <= -0.15:
            label = "BEARISH"
            color = "#FF4B4B"
        else:
            label = "NEUTRAL"
            color = "#FFAA00"

        return {
            "compound": round(compound, 3),
            "label": label,
            "color": color,
            "pos": round(pos, 3),
            "neu": round(neu, 3),
            "neg": round(neg, 3)
        }

    def _rule_based_sentiment(self, text: str) -> Tuple[float, float, float, float]:
        words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
        total_score = 0.0
        pos_count, neg_count, neu_count = 0, 0, 0

        for w in words:
            if w in FINANCIAL_LEXICON:
                score = FINANCIAL_LEXICON[w]
                total_score += score
                if score > 0:
                    pos_count += 1
                else:
                    neg_count += 1
            else:
                neu_count += 1

        total_words = max(1, len(words))
        compound = np.tanh(total_score / 3.0)
        pos = min(1.0, pos_count / total_words)
        neg = min(1.0, neg_count / total_words)
        neu = max(0.0, 1.0 - (pos + neg))
        return float(compound), float(pos), float(neu), float(neg)

    def analyze_news_feed(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyzes a batch of news articles, computes aggregate sentiment metrics and momentum.
        """
        if not articles:
            return {
                "overall_sentiment": 0.0,
                "overall_label": "NEUTRAL",
                "bullish_pct": 33.3,
                "bearish_pct": 33.3,
                "neutral_pct": 33.4,
                "analyzed_articles": []
            }

        analyzed_articles = []
        compounds = []
        counts = {"BULLISH": 0, "NEUTRAL": 0, "BEARISH": 0}

        for art in articles:
            title = art.get("title", "")
            sent = self.analyze_text(title)
            
            # Incorporate simulated sentiment if provided as ground truth
            if "simulated_sentiment" in art and self._vader_analyzer is None:
                sent["compound"] = art["simulated_sentiment"]
                if sent["compound"] > 0.1:
                    sent["label"] = "BULLISH"
                elif sent["compound"] < -0.1:
                    sent["label"] = "BEARISH"
                else:
                    sent["label"] = "NEUTRAL"

            compounds.append(sent["compound"])
            counts[sent["label"]] += 1

            enriched_art = {**art, "sentiment": sent}
            analyzed_articles.append(enriched_art)

        avg_compound = float(np.mean(compounds)) if compounds else 0.0
        total = max(1, len(articles))
        bull_pct = round((counts["BULLISH"] / total) * 100, 1)
        bear_pct = round((counts["BEARISH"] / total) * 100, 1)
        neu_pct = round((counts["NEUTRAL"] / total) * 100, 1)

        if avg_compound >= 0.15:
            overall_label = "BULLISH"
            overall_color = "#00FFA3"
        elif avg_compound <= -0.15:
            overall_label = "BEARISH"
            overall_color = "#FF4B4B"
        else:
            overall_label = "NEUTRAL"
            overall_color = "#FFAA00"

        # Word frequency extraction
        all_text = " ".join([a.get("title", "") for a in articles])
        words = [w.capitalize() for w in re.findall(r"\b[a-zA-Z]{4,}\b", all_text) if w.lower() not in ["stock", "shares", "price", "market", "after", "today", "ahead", "with", "this", "from"]]
        word_freq = pd.Series(words).value_counts().head(12).to_dict()

        return {
            "overall_sentiment": round(avg_compound, 3),
            "overall_label": overall_label,
            "overall_color": overall_color,
            "bullish_pct": bull_pct,
            "bearish_pct": bear_pct,
            "neutral_pct": neu_pct,
            "analyzed_articles": analyzed_articles,
            "top_keywords": word_freq
        }
