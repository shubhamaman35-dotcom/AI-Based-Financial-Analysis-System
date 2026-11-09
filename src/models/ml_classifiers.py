"""
Machine Learning Classifiers & Ensemble Signal Predictor Module
Trains Random Forest and Gradient Boosting algorithms on technical alpha factors.
"""

import logging
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

from src.config import ML_CONFIG
from src.models.feature_engineering import FeatureEngineer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SignalClassifier:
    """
    Supervised Machine Learning Signal Classifier predicting 5-day directional market bias.
    Blends Gradient Boosting and Random Forest with cross-validation metrics.
    """

    def __init__(self):
        self.rf_model = RandomForestClassifier(
            n_estimators=ML_CONFIG.get("n_estimators_rf", 100),
            max_depth=6,
            min_samples_split=10,
            random_state=ML_CONFIG["random_state"]
        )
        self.gb_model = HistGradientBoostingClassifier(
            max_iter=ML_CONFIG.get("n_estimators_gb", 150),
            max_depth=5,
            learning_rate=0.05,
            random_state=ML_CONFIG["random_state"]
        )
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.is_trained = False

    def train_and_evaluate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Extracts features, trains RF + GB models, computes performance metrics and feature importances.
        """
        fe = FeatureEngineer()
        X_df, y_series, feature_names = fe.prepare_feature_matrix(df)
        self.feature_names = feature_names

        if len(X_df) < 50:
            raise ValueError("Insufficient data rows for training ML classifiers (need >= 50).")

        X = X_df.values
        y = y_series.values

        # Time-series chronological train/test split (no lookahead bias)
        split_idx = int(len(X) * (1 - ML_CONFIG["test_split_ratio"]))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train models
        self.rf_model.fit(X_train_scaled, y_train)
        self.gb_model.fit(X_train_scaled, y_train)
        self.is_trained = True

        # Test set predictions (Ensemble soft voting)
        rf_prob = self.rf_model.predict_proba(X_test_scaled)[:, 1]
        gb_prob = self.gb_model.predict_proba(X_test_scaled)[:, 1]
        ensemble_prob = (0.5 * rf_prob) + (0.5 * gb_prob)
        y_pred = (ensemble_prob >= 0.5).astype(int)

        # Evaluation metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_test, ensemble_prob)
        except Exception:
            auc = 0.50

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred).tolist()

        # Feature importances from Random Forest
        importances = self.rf_model.feature_importances_
        feature_importance_df = pd.DataFrame({
            "Feature": self.feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)

        # Generate Latest Current Signal
        latest_features = fe.compute_all_features(df).iloc[-1:][self.feature_names]
        latest_scaled = self.scaler.transform(latest_features.values)
        latest_rf_prob = float(self.rf_model.predict_proba(latest_scaled)[0, 1])
        latest_gb_prob = float(self.gb_model.predict_proba(latest_scaled)[0, 1])
        latest_bullish_prob = round((latest_rf_prob + latest_gb_prob) / 2.0, 4)

        if latest_bullish_prob >= 0.65:
            signal = "STRONG BUY"
            signal_color = "green"
        elif latest_bullish_prob >= 0.53:
            signal = "BUY"
            signal_color = "lightgreen"
        elif latest_bullish_prob >= 0.45:
            signal = "HOLD / NEUTRAL"
            signal_color = "yellow"
        elif latest_bullish_prob >= 0.35:
            signal = "SELL"
            signal_color = "orange"
        else:
            signal = "STRONG SELL"
            signal_color = "red"

        return {
            "metrics": {
                "Accuracy": f"{round(acc * 100, 2)}%",
                "Precision": f"{round(prec * 100, 2)}%",
                "Recall": f"{round(rec * 100, 2)}%",
                "F1_Score": round(f1, 3),
                "ROC_AUC": round(auc, 3)
            },
            "confusion_matrix": cm,
            "feature_importance": feature_importance_df,
            "latest_signal": {
                "signal": signal,
                "color": signal_color,
                "bullish_probability": round(latest_bullish_prob * 100, 2),
                "bearish_probability": round((1 - latest_bullish_prob) * 100, 2)
            }
        }
