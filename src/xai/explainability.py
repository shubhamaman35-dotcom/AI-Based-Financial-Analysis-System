"""
Explainable AI (XAI) Module
Provides feature attribution, indicator contribution decomposition, and natural language decision explanations.
"""

import logging
from typing import Dict, List, Any
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelExplainer:
    """
    Interprets machine learning predictions to explain WHY the system generated a specific signal.
    """

    def __init__(self):
        pass

    def explain_prediction(
        self,
        feature_importance_df: pd.DataFrame,
        latest_feature_row: pd.Series,
        composite_verdict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Decomposes model prediction into top driving factors, positive vs negative contributions,
        and generates an explainable investor-facing narrative.
        """
        top_n = 8
        top_features = feature_importance_df.head(top_n).copy()
        
        explanations = []
        positive_drivers = []
        negative_drivers = []

        for _, row in top_features.iterrows():
            feat_name = row["Feature"]
            importance = float(row["Importance"])
            curr_val = float(latest_feature_row.get(feat_name, 0.0))

            # Determine indicator interpretation
            impact, note = self._interpret_indicator(feat_name, curr_val)
            
            item = {
                "Feature": feat_name,
                "Importance": round(importance * 100, 2),
                "Current_Value": round(curr_val, 4) if abs(curr_val) < 10 else round(curr_val, 2),
                "Impact": impact, # "BULLISH", "BEARISH", "NEUTRAL"
                "Note": note
            }
            explanations.append(item)

            if impact == "BULLISH":
                positive_drivers.append(f"{feat_name} ({note})")
            elif impact == "BEARISH":
                negative_drivers.append(f"{feat_name} ({note})")

        # Natural Language Summary
        verdict = composite_verdict.get("verdict", "BUY")
        score = composite_verdict.get("composite_score", 75.0)

        pos_str = ", ".join(positive_drivers[:3]) if positive_drivers else "General price stability"
        neg_str = ", ".join(negative_drivers[:2]) if negative_drivers else "Minimal downside drag"

        narrative = (
            f"The AI Engine assigned a **{verdict}** signal (Score: **{score}/100**) primarily driven by "
            f"bullish alignment in: **{pos_str}**. "
            f"Key monitored risk factors include: **{neg_str}**."
        )

        return {
            "top_drivers": pd.DataFrame(explanations),
            "narrative_explanation": narrative,
            "positive_drivers_count": len(positive_drivers),
            "negative_drivers_count": len(negative_drivers)
        }

    def _interpret_indicator(self, feat: str, val: float) -> tuple[str, str]:
        if "RSI" in feat:
            if 50 <= val <= 70:
                return "BULLISH", f"Healthy upward momentum (RSI={round(val, 1)})"
            elif val > 70:
                return "BEARISH", f"Overbought zone (RSI={round(val, 1)})"
            elif val < 30:
                return "BULLISH", f"Oversold bounce potential (RSI={round(val, 1)})"
            else:
                return "BEARISH", f"Weak momentum (RSI={round(val, 1)})"

        elif "MACD" in feat:
            if val > 0:
                return "BULLISH", f"Positive MACD spread (+{round(val, 2)})"
            else:
                return "BEARISH", f"Negative MACD momentum ({round(val, 2)})"

        elif "Dist_SMA" in feat:
            if val > 0:
                return "BULLISH", f"Trading {round(val*100, 1)}% above Moving Average"
            else:
                return "BEARISH", f"Trading {round(abs(val)*100, 1)}% below Moving Average"

        elif "BB_Pct" in feat:
            if 0.5 <= val <= 0.85:
                return "BULLISH", f"Upper Bollinger band expansion ({round(val*100, 1)}%)"
            elif val > 1.0:
                return "BEARISH", "Exceeded upper Bollinger envelope"
            else:
                return "BEARISH", f"Lower band compression ({round(val*100, 1)}%)"

        elif "Return" in feat:
            if val > 0:
                return "BULLISH", f"Positive trailing return (+{round(val*100, 2)}%)"
            else:
                return "BEARISH", f"Negative trailing return ({round(val*100, 2)}%)"

        elif "Volume_Ratio" in feat:
            if val > 1.2:
                return "BULLISH", f"High volume conviction ({round(val, 2)}x average)"
            else:
                return "NEUTRAL", f"Normal volume ({round(val, 2)}x average)"

        return "NEUTRAL", f"Value: {round(val, 2)}"
