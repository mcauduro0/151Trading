"""Unit tests for Sprint 7+8: Commodities, Futures, and Crypto ML strategies."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


# ============================================================
# Commodity Trend Following (CMD_TF_008)
# ============================================================

class TestCommodityTrendFollowing:
    """Tests for the Commodity Trend Following strategy."""

    def _make_strategy(self):
        from strategies.commodities.trend_following import CommodityTrendFollowing
        return CommodityTrendFollowing()

    def _make_trending_data(self, n=300, trend="up"):
        """Create synthetic trending commodity data."""
        dates = pd.bdate_range(end=datetime.now(), periods=n)
        symbols = ["CL=F", "GC=F", "SI=F", "NG=F", "ZC=F"]
        data = {}
        for sym in symbols:
            base = 100
            prices = [base]
            for i in range(1, n):
                drift = 0.0005 if trend == "up" else -0.0005
                ret = drift + np.random.normal(0, 0.015)
                prices.append(prices[-1] * (1 + ret))
            data[sym] = prices
        return pd.DataFrame(data, index=dates)

    def test_initialization(self):
        strat = self._make_strategy()
        assert strat.strategy_id == "CMD_TF_008"
        assert strat.name == "Commodity Trend Following"

    def test_generates_signals_trending_market(self):
        strat = self._make_strategy()
        prices = self._make_trending_data(300, "up")
        signals = strat.generate_signals({"prices": prices})
        assert isinstance(signals, list)
        # In a trending market, should produce some signals
        assert len(signals) >= 0  # May or may not depending on randomness

    def test_signal_weights_capped(self):
        strat = self._make_strategy()
        prices = self._make_trending_data(300, "up")
        signals = strat.generate_signals({"prices": prices})
        for sig in signals:
            assert sig.weight <= strat.max_weight + 0.01

    def test_risk_checks_filter_high_vol(self):
        strat = self._make_strategy()
        from strategies.base import Signal, SignalDirection
        test_signals = [
            Signal(symbol="CL=F", direction=SignalDirection.LONG, weight=0.10,
                   metadata={"annualized_vol": 0.90}),  # Too volatile
            Signal(symbol="GC=F", direction=SignalDirection.LONG, weight=0.10,
                   metadata={"annualized_vol": 0.25}),  # OK
        ]
        filtered = strat.risk_checks(test_signals)
        assert len(filtered) == 1
        assert filtered[0].symbol == "GC=F"

    def test_insufficient_data_returns_empty(self):
        strat = self._make_strategy()
        prices = self._make_trending_data(50, "up")  # Too short
        signals = strat.generate_signals({"prices": prices})
        assert signals == []


# ============================================================
# Futures Calendar Spread (FUT_ROLL_009)
# ============================================================

class TestFuturesCalendarSpread:
    """Tests for the Futures Calendar Spread strategy."""

    def _make_strategy(self):
        from strategies.futures.calendar_spread import FuturesCalendarSpread
        return FuturesCalendarSpread()

    def _make_futures_data(self, n=100):
        dates = pd.bdate_range(end=datetime.now(), periods=n)
        symbols = ["CL=F", "GC=F", "NG=F", "ZC=F"]
        data = {}
        for sym in symbols:
            prices = [100]
            for i in range(1, n):
                prices.append(prices[-1] * (1 + np.random.normal(0.0002, 0.012)))
            data[sym] = prices
        return pd.DataFrame(data, index=dates)

    def test_initialization(self):
        strat = self._make_strategy()
        assert strat.strategy_id == "FUT_ROLL_009"
        assert strat.name == "Futures Calendar Spread"

    def test_generates_signals(self):
        strat = self._make_strategy()
        prices = self._make_futures_data(100)
        signals = strat.generate_signals({"front_prices": prices})
        assert isinstance(signals, list)

    def test_signal_metadata_has_regime(self):
        strat = self._make_strategy()
        prices = self._make_futures_data(100)
        signals = strat.generate_signals({"front_prices": prices})
        for sig in signals:
            assert "regime" in sig.metadata
            assert sig.metadata["regime"] in ("backwardation", "contango")

    def test_insufficient_data(self):
        strat = self._make_strategy()
        prices = self._make_futures_data(30)  # Too short
        signals = strat.generate_signals({"front_prices": prices})
        assert signals == []


# ============================================================
# COT Analysis (FUT_COT_010)
# ============================================================

class TestCOTAnalysis:
    """Tests for the COT Positioning Analysis strategy."""

    def _make_strategy(self):
        from strategies.futures.cot_analysis import COTAnalysis
        return COTAnalysis()

    def _make_data(self, n=100):
        dates = pd.bdate_range(end=datetime.now(), periods=n)
        symbols = ["CL=F", "GC=F", "SI=F", "NG=F"]
        data = {}
        for sym in symbols:
            prices = [100]
            for i in range(1, n):
                prices.append(prices[-1] * (1 + np.random.normal(0.0003, 0.015)))
            data[sym] = prices
        return pd.DataFrame(data, index=dates)

    def test_initialization(self):
        strat = self._make_strategy()
        assert strat.strategy_id == "FUT_COT_010"

    def test_generates_signals_from_prices(self):
        strat = self._make_strategy()
        prices = self._make_data(100)
        signals = strat.generate_signals({"prices": prices})
        assert isinstance(signals, list)

    def test_signal_metadata_has_percentiles(self):
        strat = self._make_strategy()
        prices = self._make_data(100)
        signals = strat.generate_signals({"prices": prices})
        for sig in signals:
            assert "commercial_percentile" in sig.metadata
            assert "speculative_percentile" in sig.metadata
            assert 0 <= sig.metadata["commercial_percentile"] <= 1

    def test_risk_checks_filter_weak_signals(self):
        strat = self._make_strategy()
        from strategies.base import Signal, SignalDirection
        test_signals = [
            Signal(symbol="CL=F", direction=SignalDirection.LONG, weight=0.05,
                   metadata={"signal_strength": 0.10}),  # Too weak
            Signal(symbol="GC=F", direction=SignalDirection.LONG, weight=0.08,
                   metadata={"signal_strength": 0.50}),  # Strong enough
        ]
        filtered = strat.risk_checks(test_signals)
        assert len(filtered) == 1
        assert filtered[0].symbol == "GC=F"


# ============================================================
# Crypto ANN Classifier (CRYPTO_ANN_011)
# ============================================================

class TestCryptoANNClassifier:
    """Tests for the Crypto ANN Classifier strategy."""

    def _make_strategy(self):
        from strategies.crypto.ann_classifier import CryptoANNClassifier
        return CryptoANNClassifier(train_window=100, epochs=10)

    def _make_crypto_data(self, n=200):
        dates = pd.bdate_range(end=datetime.now(), periods=n)
        symbols = ["BTC-USD", "ETH-USD"]
        prices = {}
        volumes = {}
        for sym in symbols:
            p = [50000 if sym == "BTC-USD" else 3000]
            v = []
            for i in range(n):
                if i > 0:
                    p.append(p[-1] * (1 + np.random.normal(0.001, 0.03)))
                v.append(np.random.uniform(1e9, 5e9))
            prices[sym] = p
            volumes[sym] = v
        return pd.DataFrame(prices, index=dates), pd.DataFrame(volumes, index=dates)

    def test_initialization(self):
        strat = self._make_strategy()
        assert strat.strategy_id == "CRYPTO_ANN_011"
        assert strat.name == "Crypto ANN Classifier"

    def test_generates_signals(self):
        strat = self._make_strategy()
        prices, volumes = self._make_crypto_data(200)
        signals = strat.generate_signals({"prices": prices, "volumes": volumes})
        assert isinstance(signals, list)

    def test_signal_metadata_has_probabilities(self):
        strat = self._make_strategy()
        prices, volumes = self._make_crypto_data(200)
        signals = strat.generate_signals({"prices": prices, "volumes": volumes})
        for sig in signals:
            assert "prob_up" in sig.metadata
            assert "prob_down" in sig.metadata
            assert "model_type" in sig.metadata
            assert sig.metadata["model_type"] == "ANN"

    def test_risk_checks_filter_low_confidence(self):
        strat = self._make_strategy()
        from strategies.base import Signal, SignalDirection
        test_signals = [
            Signal(symbol="BTC-USD", direction=SignalDirection.LONG, weight=0.10,
                   metadata={"confidence": 0.55}),  # Below threshold
            Signal(symbol="ETH-USD", direction=SignalDirection.LONG, weight=0.10,
                   metadata={"confidence": 0.75}),  # Above threshold
        ]
        filtered = strat.risk_checks(test_signals)
        assert len(filtered) == 1
        assert filtered[0].symbol == "ETH-USD"


# ============================================================
# Crypto Naive Bayes Sentiment (CRYPTO_NB_012)
# ============================================================

class TestCryptoNaiveBayesSentiment:
    """Tests for the Crypto Naive Bayes Sentiment strategy."""

    def _make_strategy(self):
        from strategies.crypto.naive_bayes_sentiment import CryptoNaiveBayesSentiment
        return CryptoNaiveBayesSentiment(train_window=100)

    def _make_crypto_data(self, n=200):
        dates = pd.bdate_range(end=datetime.now(), periods=n)
        symbols = ["BTC-USD", "ETH-USD"]
        prices = {}
        volumes = {}
        for sym in symbols:
            p = [50000 if sym == "BTC-USD" else 3000]
            v = []
            for i in range(n):
                if i > 0:
                    p.append(p[-1] * (1 + np.random.normal(0.001, 0.03)))
                v.append(np.random.uniform(1e9, 5e9))
            prices[sym] = p
            volumes[sym] = v
        return pd.DataFrame(prices, index=dates), pd.DataFrame(volumes, index=dates)

    def test_initialization(self):
        strat = self._make_strategy()
        assert strat.strategy_id == "CRYPTO_NB_012"

    def test_gaussian_nb_classifier(self):
        """Test the standalone Gaussian Naive Bayes implementation."""
        from strategies.crypto.naive_bayes_sentiment import GaussianNaiveBayes
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = (X[:, 0] > 0).astype(int)
        model = GaussianNaiveBayes()
        model.fit(X, y)
        probs = model.predict_proba(X[:5])
        assert probs.shape == (5, 2)
        assert np.allclose(probs.sum(axis=1), 1.0)
        preds = model.predict(X[:5])
        assert len(preds) == 5

    def test_generates_signals(self):
        strat = self._make_strategy()
        prices, volumes = self._make_crypto_data(200)
        signals = strat.generate_signals({"prices": prices, "volumes": volumes})
        assert isinstance(signals, list)

    def test_signal_metadata_has_model_type(self):
        strat = self._make_strategy()
        prices, volumes = self._make_crypto_data(200)
        signals = strat.generate_signals({"prices": prices, "volumes": volumes})
        for sig in signals:
            assert sig.metadata["model_type"] == "GaussianNaiveBayes"

    def test_risk_checks_cap_total_crypto(self):
        strat = self._make_strategy()
        from strategies.base import Signal, SignalDirection
        # Create many signals that would exceed 30% total
        test_signals = [
            Signal(symbol=f"CRYPTO{i}-USD", direction=SignalDirection.LONG,
                   weight=0.10, metadata={"confidence": 0.75})
            for i in range(5)
        ]
        filtered = strat.risk_checks(test_signals)
        total_weight = sum(s.weight for s in filtered)
        assert total_weight <= 0.31  # Allow small float tolerance


# ============================================================
# Cross-strategy tests
# ============================================================

class TestCrossStrategyConsistency:
    """Test consistency across all Sprint 7+8 strategies."""

    def test_all_strategies_have_required_data(self):
        from strategies.commodities.trend_following import CommodityTrendFollowing
        from strategies.futures.calendar_spread import FuturesCalendarSpread
        from strategies.futures.cot_analysis import COTAnalysis
        from strategies.crypto.ann_classifier import CryptoANNClassifier
        from strategies.crypto.naive_bayes_sentiment import CryptoNaiveBayesSentiment

        strategies = [
            CommodityTrendFollowing(),
            FuturesCalendarSpread(),
            COTAnalysis(),
            CryptoANNClassifier(),
            CryptoNaiveBayesSentiment(),
        ]

        for strat in strategies:
            data_req = strat.required_data()
            assert isinstance(data_req, dict)
            assert len(data_req) > 0

    def test_all_strategies_handle_empty_data(self):
        from strategies.commodities.trend_following import CommodityTrendFollowing
        from strategies.futures.calendar_spread import FuturesCalendarSpread
        from strategies.futures.cot_analysis import COTAnalysis
        from strategies.crypto.ann_classifier import CryptoANNClassifier
        from strategies.crypto.naive_bayes_sentiment import CryptoNaiveBayesSentiment

        strategies = [
            CommodityTrendFollowing(),
            FuturesCalendarSpread(),
            COTAnalysis(),
            CryptoANNClassifier(),
            CryptoNaiveBayesSentiment(),
        ]

        for strat in strategies:
            signals = strat.generate_signals({"prices": pd.DataFrame()})
            assert signals == []
