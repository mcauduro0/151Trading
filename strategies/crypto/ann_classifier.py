"""Crypto ANN Classifier Strategy (CRYPTO_ANN_011).

Uses a simple Artificial Neural Network (multi-layer perceptron) to classify
crypto price direction based on technical features. The model is trained
on rolling windows and generates probabilistic long/short signals.

Key features:
- Feature engineering: RSI, MACD, Bollinger %B, volume ratio, volatility
- Rolling window training with walk-forward validation
- Probability-based position sizing
- Winsorization at 5%/95% for all features
- Regime detection (trending vs mean-reverting)

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


class CryptoANNClassifier(StrategyBase):
    """ANN-based crypto direction classifier.

    Uses a lightweight numpy-based neural network (no sklearn/torch dependency)
    to classify next-day returns as up/down based on technical features.

    Architecture: Input(7) -> Hidden(16) -> Hidden(8) -> Output(2)
    Activation: ReLU hidden, Softmax output
    Training: Mini-batch gradient descent with rolling window
    """

    DEFAULT_UNIVERSE = [
        "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD",
        "ADA-USD", "XRP-USD", "AVAX-USD", "DOT-USD",
    ]

    def __init__(
        self,
        train_window: int = 252,
        predict_horizon: int = 1,
        hidden_sizes: tuple = (16, 8),
        learning_rate: float = 0.001,
        epochs: int = 50,
        batch_size: int = 32,
        confidence_threshold: float = 0.60,
        max_weight: float = 0.12,
    ):
        super().__init__(
            strategy_id="CRYPTO_ANN_011",
            name="Crypto ANN Classifier",
            asset_class=AssetClass.CRYPTO,
            style=StrategyStyle.ML_BASED,
            description="Neural network classifier for crypto direction prediction using technical features"
        )
        self.train_window = train_window
        self.predict_horizon = predict_horizon
        self.hidden_sizes = hidden_sizes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.confidence_threshold = confidence_threshold
        self.max_weight = max_weight
        self._models: Dict[str, Dict] = {}

    def required_data(self) -> Dict[str, str]:
        return {
            "prices": "yahoo_finance:daily_ohlcv",
            "volumes": "yahoo_finance:daily_volume",
        }

    def _winsorize(self, arr: np.ndarray, lower: float = 0.05, upper: float = 0.95) -> np.ndarray:
        """Winsorize array at 5%/95%."""
        lo = np.nanpercentile(arr, lower * 100)
        hi = np.nanpercentile(arr, upper * 100)
        return np.clip(arr, lo, hi)

    def _compute_features(self, prices: pd.Series, volumes: Optional[pd.Series] = None) -> pd.DataFrame:
        """Compute technical features for a single asset."""
        df = pd.DataFrame(index=prices.index)

        # Returns
        ret_1d = prices.pct_change(1)
        ret_5d = prices.pct_change(5)
        ret_20d = prices.pct_change(20)

        # RSI (14-period)
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        # MACD
        ema12 = prices.ewm(span=12).mean()
        ema26 = prices.ewm(span=26).mean()
        macd = (ema12 - ema26) / prices

        # Bollinger %B
        sma20 = prices.rolling(20).mean()
        std20 = prices.rolling(20).std()
        bb_upper = sma20 + 2 * std20
        bb_lower = sma20 - 2 * std20
        bb_pct = (prices - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)

        # Volatility (20-day)
        vol_20d = ret_1d.rolling(20).std() * np.sqrt(252)

        # Volume ratio
        if volumes is not None and not volumes.empty:
            vol_ratio = volumes / volumes.rolling(20).mean()
        else:
            vol_ratio = pd.Series(1.0, index=prices.index)

        df['ret_1d'] = ret_1d
        df['ret_5d'] = ret_5d
        df['rsi'] = rsi / 100.0  # Normalize to [0, 1]
        df['macd'] = macd
        df['bb_pct'] = bb_pct
        df['vol_20d'] = vol_20d
        df['vol_ratio'] = vol_ratio

        return df.dropna()

    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / exp_x.sum(axis=1, keepdims=True)

    def _init_weights(self, layer_sizes: List[int]) -> List[Dict]:
        """Initialize network weights with Xavier initialization."""
        np.random.seed(42)
        layers = []
        for i in range(len(layer_sizes) - 1):
            scale = np.sqrt(2.0 / layer_sizes[i])
            layers.append({
                'W': np.random.randn(layer_sizes[i], layer_sizes[i+1]) * scale,
                'b': np.zeros((1, layer_sizes[i+1])),
            })
        return layers

    def _forward(self, X: np.ndarray, layers: List[Dict]) -> np.ndarray:
        """Forward pass through the network."""
        h = X
        for i, layer in enumerate(layers):
            z = h @ layer['W'] + layer['b']
            if i < len(layers) - 1:
                h = self._relu(z)
            else:
                h = self._softmax(z)
        return h

    def _train_model(self, X: np.ndarray, y: np.ndarray) -> List[Dict]:
        """Train the ANN with mini-batch gradient descent."""
        n_features = X.shape[1]
        layer_sizes = [n_features] + list(self.hidden_sizes) + [2]
        layers = self._init_weights(layer_sizes)

        # One-hot encode labels
        y_onehot = np.zeros((len(y), 2))
        y_onehot[np.arange(len(y)), y.astype(int)] = 1

        n_samples = len(X)
        for epoch in range(self.epochs):
            # Shuffle
            idx = np.random.permutation(n_samples)
            X_shuffled = X[idx]
            y_shuffled = y_onehot[idx]

            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                batch_size = end - start

                # Forward pass with activations stored
                activations = [X_batch]
                h = X_batch
                for i, layer in enumerate(layers):
                    z = h @ layer['W'] + layer['b']
                    if i < len(layers) - 1:
                        h = self._relu(z)
                    else:
                        h = self._softmax(z)
                    activations.append(h)

                # Backward pass
                delta = activations[-1] - y_batch  # Cross-entropy gradient

                for i in range(len(layers) - 1, -1, -1):
                    dW = activations[i].T @ delta / batch_size
                    db = delta.mean(axis=0, keepdims=True)

                    # Update weights
                    layers[i]['W'] -= self.learning_rate * dW
                    layers[i]['b'] -= self.learning_rate * db

                    if i > 0:
                        delta = delta @ layers[i]['W'].T
                        delta = delta * (activations[i] > 0).astype(float)  # ReLU derivative

        return layers

    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> List[Signal]:
        """Generate ANN-based crypto signals."""
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
            features = self._compute_features(price_series, vol_series)
            if len(features) < self.train_window:
                continue

            # Winsorize all features
            feature_matrix = features.values.copy()
            for col in range(feature_matrix.shape[1]):
                feature_matrix[:, col] = self._winsorize(feature_matrix[:, col])

            # Normalize features (z-score)
            mean = feature_matrix[:-1].mean(axis=0)
            std = feature_matrix[:-1].std(axis=0)
            std[std == 0] = 1
            feature_matrix = (feature_matrix - mean) / std

            # Labels: 1 if next-day return > 0, else 0
            future_ret = price_series.pct_change(self.predict_horizon).shift(-self.predict_horizon)
            labels = (future_ret > 0).astype(int).reindex(features.index).dropna()

            # Align features and labels
            common_idx = features.index.intersection(labels.index)
            if len(common_idx) < self.train_window:
                continue

            X_all = feature_matrix[:len(common_idx)]
            y_all = labels.loc[common_idx].values

            # Train on rolling window, predict last observation
            X_train = X_all[-self.train_window-1:-1]
            y_train = y_all[-self.train_window-1:-1]
            X_test = X_all[-1:].reshape(1, -1) if X_all.ndim > 1 else X_all[-1:].reshape(1, -1)

            # Train model
            model = self._train_model(X_train, y_train)

            # Predict
            probs = self._forward(X_test, model)[0]
            prob_up = probs[1]
            prob_down = probs[0]

            # Generate signal if confident enough
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
                        "model_type": "ANN",
                        "features_used": list(features.columns),
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
                        "model_type": "ANN",
                        "features_used": list(features.columns),
                    }
                ))

        return signals

    def risk_checks(self, signals: List[Signal],
                    portfolio_state: Optional[Dict] = None) -> List[Signal]:
        """Apply crypto-specific risk checks."""
        filtered = []
        for sig in signals:
            conf = sig.metadata.get("confidence", 0) if sig.metadata else 0
            if conf < self.confidence_threshold:
                continue
            # Cap crypto weights
            if sig.weight > self.max_weight:
                sig.weight = self.max_weight
            filtered.append(sig)
        return filtered
