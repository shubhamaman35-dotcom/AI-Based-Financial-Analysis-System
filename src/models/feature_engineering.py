"""
Feature Engineering Module
Calculates 30+ comprehensive quantitative and technical indicators from raw OHLCV series.
Pure Python & Vectorized NumPy/Pandas implementation (zero external TA-Lib C dependencies).
"""

import logging
from typing import Tuple, List, Dict
import numpy as np
import pandas as pd

from src.config import INDICATOR_PARAMS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Transforms raw OHLCV data into high-dimensional alpha features,
    technical indicators, momentum oscillators, volatility metrics, and targets.
    """

    def __init__(self, params: Dict = INDICATOR_PARAMS):
        self.params = params

    def compute_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes all technical features and enriches dataframe.
        """
        df = df.copy()
        
        # Ensure standard column names
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required price column: {col}")

        # 1. Moving Averages & Trend Indicators
        df = self._add_trend_indicators(df)

        # 2. Momentum & Oscillators
        df = self._add_momentum_oscillators(df)

        # 3. Volatility Indicators
        df = self._add_volatility_indicators(df)

        # 4. Volume Indicators
        df = self._add_volume_indicators(df)

        # 5. Price Action & Statistical Features
        df = self._add_statistical_features(df)

        # Clean NaNs resulting from rolling windows
        df = df.bfill().ffill()

        return df

    def _add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        # Simple Moving Averages
        df["SMA_20"] = df["Close"].rolling(window=self.params.get("sma_short", 20)).mean()
        df["SMA_50"] = df["Close"].rolling(window=self.params.get("sma_medium", 50)).mean()
        df["SMA_200"] = df["Close"].rolling(window=self.params.get("sma_long", 200)).mean()

        # Exponential Moving Averages
        df["EMA_9"] = df["Close"].ewm(span=9, adjust=False).mean()
        df["EMA_21"] = df["Close"].ewm(span=21, adjust=False).mean()
        df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()

        # Distance from Moving Averages (%)
        df["Dist_SMA_20"] = (df["Close"] - df["SMA_20"]) / df["SMA_20"]
        df["Dist_SMA_50"] = (df["Close"] - df["SMA_50"]) / df["SMA_50"]
        df["Dist_SMA_200"] = (df["Close"] - df["SMA_200"]) / df["SMA_200"]

        # MACD (Moving Average Convergence Divergence)
        fast_ema = df["Close"].ewm(span=self.params.get("macd_fast", 12), adjust=False).mean()
        slow_ema = df["Close"].ewm(span=self.params.get("macd_slow", 26), adjust=False).mean()
        df["MACD"] = fast_ema - slow_ema
        df["MACD_Signal"] = df["MACD"].ewm(span=self.params.get("macd_signal", 9), adjust=False).mean()
        df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

        return df

    def _add_momentum_oscillators(self, df: pd.DataFrame) -> pd.DataFrame:
        # Relative Strength Index (RSI)
        delta = df["Close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        rsi_period = self.params.get("rsi_period", 14)
        avg_gain = gain.ewm(com=rsi_period - 1, min_periods=rsi_period).mean()
        avg_loss = loss.ewm(com=rsi_period - 1, min_periods=rsi_period).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        df["RSI"] = 100 - (100 / (1 + rs))

        # Stochastic Oscillator (%K and %D)
        stoch_k_period = self.params.get("stoch_k", 14)
        low_min = df["Low"].rolling(window=stoch_k_period).min()
        high_max = df["High"].rolling(window=stoch_k_period).max()
        df["Stoch_K"] = 100 * ((df["Close"] - low_min) / ((high_max - low_min) + 1e-10))
        df["Stoch_D"] = df["Stoch_K"].rolling(window=self.params.get("stoch_d", 3)).mean()

        # Williams %R
        williams_period = self.params.get("williams_period", 14)
        highest_high = df["High"].rolling(window=williams_period).max()
        lowest_low = df["Low"].rolling(window=williams_period).min()
        df["Williams_R"] = -100 * ((highest_high - df["Close"]) / ((highest_high - lowest_low) + 1e-10))

        # Rate of Change (ROC - 10 day)
        df["ROC_10"] = df["Close"].pct_change(periods=10) * 100

        # Commodity Channel Index (CCI)
        typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
        tp_sma = typical_price.rolling(window=20).mean()
        mean_dev = typical_price.rolling(window=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
        df["CCI"] = (typical_price - tp_sma) / (0.015 * (mean_dev + 1e-10))

        return df

    def _add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        # Bollinger Bands
        bb_period = self.params.get("bb_period", 20)
        bb_std_mult = self.params.get("bb_std", 2.0)
        
        df["BB_Middle"] = df["Close"].rolling(window=bb_period).mean()
        bb_std = df["Close"].rolling(window=bb_period).std()
        df["BB_Upper"] = df["BB_Middle"] + (bb_std * bb_std_mult)
        df["BB_Lower"] = df["BB_Middle"] - (bb_std * bb_std_mult)
        df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / (df["BB_Middle"] + 1e-10)
        df["BB_Pct"] = (df["Close"] - df["BB_Lower"]) / ((df["BB_Upper"] - df["BB_Lower"]) + 1e-10)

        # Average True Range (ATR)
        prev_close = df["Close"].shift(1)
        tr1 = df["High"] - df["Low"]
        tr2 = (df["High"] - prev_close).abs()
        tr3 = (df["Low"] - prev_close).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["ATR"] = true_range.rolling(window=self.params.get("atr_period", 14)).mean()
        df["ATR_Pct"] = df["ATR"] / df["Close"]

        # Realized Historical Volatility (20-day annualized)
        df["Hist_Vol_20"] = df["Close"].pct_change().rolling(window=20).std() * np.sqrt(252)

        return df

    def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        # On-Balance Volume (OBV)
        direction = np.sign(df["Close"].diff()).fillna(0)
        df["OBV"] = (direction * df["Volume"]).cumsum()

        # Volume Moving Average & Volume Ratio
        df["Volume_SMA_20"] = df["Volume"].rolling(window=20).mean()
        df["Volume_Ratio"] = df["Volume"] / (df["Volume_SMA_20"] + 1e-10)

        # Approximate Volume Weighted Average Price (VWAP - 20 day rolling)
        cum_vol_price = (df["Close"] * df["Volume"]).rolling(window=20).sum()
        cum_vol = df["Volume"].rolling(window=20).sum()
        df["VWAP_20"] = cum_vol_price / (cum_vol + 1e-10)

        return df

    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Returns & Lag Returns
        df["Return_1d"] = df["Close"].pct_change(1)
        df["Return_2d"] = df["Close"].pct_change(2)
        df["Return_5d"] = df["Close"].pct_change(5)
        df["Return_10d"] = df["Close"].pct_change(10)
        df["Return_21d"] = df["Close"].pct_change(21)

        # High-Low Range Spread & Close-Open Gap
        df["HL_Spread"] = (df["High"] - df["Low"]) / df["Close"]
        df["CO_Spread"] = (df["Close"] - df["Open"]) / df["Open"]

        # Target Variables for ML Modeling
        # 1. Forward 1-day & 5-day return
        df["Target_Return_1d"] = df["Close"].pct_change(1).shift(-1)
        df["Target_Return_5d"] = df["Close"].pct_change(5).shift(-5)
        
        # 2. Binary Directional Target: 1 if Forward 5-day return > 0.5%, else 0
        df["Target_Direction"] = np.where(df["Target_Return_5d"] > 0.005, 1, 0)
        
        # 3. Multi-Class Signal: 1 (Buy if > 1.5%), -1 (Sell if < -1.5%), 0 (Hold)
        df["Target_Signal"] = 0
        df.loc[df["Target_Return_5d"] > 0.015, "Target_Signal"] = 1
        df.loc[df["Target_Return_5d"] < -0.015, "Target_Signal"] = -1

        return df

    def prepare_feature_matrix(
        self,
        df: pd.DataFrame,
        feature_cols: List[str] = None
    ) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Extracts clean feature matrix X and target y, dropping lookahead rows at the end.
        """
        df_feat = self.compute_all_features(df)
        
        if feature_cols is None:
            # Drop raw price, date indices, and target columns from X
            exclude_cols = [
                "Target_Return_1d", "Target_Return_5d", "Target_Direction", "Target_Signal",
                "Open", "High", "Low", "Close", "Volume"
            ]
            feature_cols = [c for c in df_feat.columns if c not in exclude_cols]

        # Drop rows with NaN targets (the final lookahead rows)
        clean_df = df_feat.dropna(subset=["Target_Direction", "Target_Return_5d"])
        
        X = clean_df[feature_cols]
        y = clean_df["Target_Direction"]
        
        return X, y, feature_cols
