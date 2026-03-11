"""Crypto Naive Bayes Sentiment Strategy (CRYPTO_NB_012).

Uses a Naive Bayes classifier on sentiment-derived features to predict
crypto price direction. Combines social media sentiment scores with
technical indicators for a hybrid ML approach.

Key features:
- Gaussian Naive Bayes classifier (numpy-based, no sklearn dependency)
- Sentiment features from Reddit/social media volume proxies
- Technical feature overlay (RSI, momentum, volume)
- Walk-forward validation with rolling training window
- Winsorization at 5%/95% for all features

Reference: 151 Trading Strategies, Crypto ML section.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from strategies.base import (
    StrategyBase, Signal, SignalDirection, AssetClass, StrategyStyle
)


class GaussianNaiveBayes:
    """Lightweight Gaussian Naive Bayes classifier (no sklearn dependency)."""

    def __init__(self):
        self.class_priors: Optional[np.ndarray] = None
        self.means: Optional[np.ndarray] = None
        self.variances: Optional[np.ndarray] = None
        self.classes: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GaussianNaiveBayes':
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        n_features = X.shape[1]

        self.means = np.zeros((n_classes, n_features))
        self.variances = np.zeros((n_classes, n_features))
        self.class_priors = np.zeros(n_classes)

        for i, c in enumerate(self.classes):
            X_c = X[y == c]
            self.means[i] = X_c.mean(axis=0)
            self.variances[i] = X_c.var(axis=0) + 1e-9  # Smoothing
            self.class_priors[i] = len(X_c) / len(X)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        n_samples = X.shape[0]
        n_classes = len(self.classes)
        log_probs = np.zeros((n_samples, n_classes))

        for i in range(n_classes):
            log_prior = np.log(self.class_priors[i])
            log_likelihood = -0.5 * np.sum(
                np.log(2 * np.pi * self.variances[i]) +
                (X - self.means[i]) ** 2 / self.variances[i],
                axis=1
            )
            log_probs[:, i] = log_prior + log_likelihood

        # Convert to probabilities via softmax
        log_probs -= log_probs.max(axis=1, keepdims=True)
        probs = np.exp(log_probs)
        probs /= probs.sum(axis=1, keepdims=True)
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return self.classes[np.argmax(probs, axis=1)]


class CryptoNaiveBayesSentiment(StrategyBase):
    """Naive Bayes sentiment-based crypto strategy.

    Combines sentiment-derived features with technical indicators
    using a Gaussian Naive Bayes classifier for direction prediction.
    """

    DEFAULT_UNIVERSE = [
        "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD",
        "ADA-USD", "XRP-USD", "DOGE-USD", "AVAX-USD",
    ]

    def __init__(
        self,
        train_window: int = 180,
        sentiment_lookback: int = 7,
        confidence_threshold: float = 0.60,
        max_weight: float = 0.10,
    ):
        super().__init__(
            strategy_id="CRYPTO_NB_012",
            name="Crypto Naive Bayes Sentiment",
            asset_class=AssetClass.CRYPTO,
            style=StrategyStyle.ML_BASED,
            description="Gaussian Naive Bayes classifier combining sentiment and technical features for crypto"
        )
        self.train_window = train_window
        self.sentiment_lookback = sentiment_lookback
        self.confidence_threshold = confidence_threshold
        self.max_weight = max_weight

    def required_data(self) -> Dict[str, str]:
        return {
            "prices": "yahoo_finance:daily_ohlcv",
            "volumes": "yahoo_finance:daily_volume",
            "sentiment": "reddit:crypto_sentiment",
        }

    def _winsorize(self, arr: np.ndarray, lower: float = 0.05, upper: float = 0.95) -> np.ndarray:
        """Winsorize at 5%/95%."""
        lo = np.nanpercentile(arr, lower * 100)
        hi = np.nanpercentile(arr, upper * 100)
        return np.clip(arr, lo, hi)

    def _compute_sentiment_features(self, prices: pd.Series, volumes: Optional[pd.Series] = None) -> pd.DataFrame:
        """Compute sentiment proxy features from price/volume patterns.

        In production, these would be replaced with actual Reddit/Twitter
        sentiment scores. Here we use volume surges and price momentum
        as sentiment proxies.
        """
        df = pd.DataFrame(index=prices.index)

        # Volume surge as sentiment proxy (high volume = high attention)
        if volumes is not None and not volumes.empty:
            vol_ma = volumes.rolling(20).mean()
            df['volume_surge'] = (volumes / vol_ma.replace(0, np.nan)).fillna(1.0)
            df['volume_trend'] = volumes.rolling(self.sentiment_lookback).mean() / vol_ma
        else:
            df['volume_surge'] = 1.0
            df['volume_trend'] = 1.0

        # Price momentum as "market sentiment"
        df['momentum_7d'] = prices.pct_change(7)
        df['momentum_30d'] = prices.pct_change(30)

        # Volatility regime (high vol = fear, low vol = complacency)
        ret = prices.pct_change()
        df['realized_vol'] = ret.rolling(20).std() * np.sqrt(252)
        df['vol_regime'] = df['realized_vol'] / df['realized_vol'].rolling(60).mean()

        # RSI as sentiment gauge
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df['rsi'] = (100 - (100 / (1 + rs))) / 100.0

        # Mean reversion signal
        sma_20 = prices.rolling(20).mean()
        df['price_vs_sma'] = (prices - sma_20) / sma_20

        return df.dropna()

    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> List[Signal]:
        """Generate Naive Bayes sentiment signals for crypto."""
        prices = data.get("prices")
        if prices is None or prices.empty:
            return []

        volumes = data.get("volumes")
        signals = []

        for symbol in prices.columns:
            price_series = prices[symbol].dropna()
            if len(price_series) < self.train_window + 30:
                continue

            vol_series = volumes[symbol] if volumes is not None and symbol in volumes.columns else None

            # Compute features
            features = self._compute_sentiment_features(price_series, vol_series)
            if len(features) < self.train_window:
                continue

            # Winsorize all features
            feature_matrix = features.values.copy()
            for col in range(feature_matrix.shape[1]):
                feature_matrix[:, col] = self._winsorize(feature_matrix[:, col])

            # Labels: 1 if next-day return > 0, else 0
            future_ret = price_series.pct_change(1).shift(-1)
            labels = (future_ret > 0).astype(int).reindex(features.index).dropna()

            common_idx = features.index.intersection(labels.index)
            if len(common_idx) < self.train_window:
                continue

            X_all = feature_matrix[:len(common_idx)]
            y_all = labels.loc[common_idx].values

            # Train on rolling window
            X_train = X_all[-self.train_window-1:-1]
            y_train = y_all[-self.train_window-1:-1]
            X_test = X_all[-1:].reshape(1, -1)

            # Train Naive Bayes
            model = GaussianNaiveBayes()
            model.fit(X_train, y_train)

            # Predict
            probs = model.predict_proba(X_test)[0]

            # Map classes to probabilities
            if len(model.classes) < 2:
                continue

            idx_up = np.where(model.classes == 1)[0]
            idx_down = np.where(model.classes == 0)[0]

            prob_up = probs[idx_up[0]] if len(idx_up) > 0 else 0.5
            prob_down = probs[idx_down[0]] if len(idx_down) > 0 else 0.5

            # Current feature values for metadata
            current_features = {col: float(features[col].iloc[-1]) for col in features.columns}

            # Generate signal
            if prob_up > self.confidence_threshold:
                weight = min((prob_up - 0.5) * 2 * self.max_weight, self.max_weight)
                signals.append(Signal(
                    symbol=symbol,
                    direction=SignalDirection.LONG,
                    weight=weight,
                    metadata={
                        "prob_up": float(prob_up),
                        "prob_down": float(prob_down),
                        "confidence": float(prob_up),
                        "model_type": "GaussianNaiveBayes",
                        "rsi": current_features.get("rsi", 0),
                        "volume_surge": current_features.get("volume_surge", 0),
                        "vol_regime": current_features.get("vol_regime", 0),
                    }
                ))
            elif prob_down > self.confidence_threshold:
                weight = min((prob_down - 0.5) * 2 * self.max_weight, self.max_weight)
                signals.append(Signal(
                    symbol=symbol,
                    direction=SignalDirection.SHORT,
                    weight=weight,
                    metadata={
                        "prob_up": float(prob_up),
                        "prob_down": float(prob_down),
                        "confidence": float(prob_down),
                        "model_type": "GaussianNaiveBayes",
                        "rsi": current_features.get("rsi", 0),
                        "volume_surge": current_features.get("volume_surge", 0),
                        "vol_regime": current_features.get("vol_regime", 0),
                    }
                ))

        return signals

    def risk_checks(self, signals: List[Signal],
                    portfolio_state: Optional[Dict] = None) -> List[Signal]:
        """Apply crypto-specific risk checks."""
        filtered = []
        total_weight = 0
        for sig in signals:
            conf = sig.metadata.get("confidence", 0) if sig.metadata else 0
            if conf < self.confidence_threshold:
                continue
            # Cap individual and total crypto exposure
            if sig.weight > self.max_weight:
                sig.weight = self.max_weight
            if total_weight + sig.weight > 0.30:  # Max 30% total crypto
                sig.weight = max(0, 0.30 - total_weight)
            if sig.weight > 0.01:
                total_weight += sig.weight
                filtered.append(sig)
        return filtered
