"""Unit tests for the Strategy Abstract Base Class."""

import pytest
import pandas as pd
import numpy as np
from datetime import date
from typing import Any, Dict

import sys
sys.path.insert(0, "/home/ubuntu/151Trading")

from strategies.base import (
    BaseStrategy, StrategyMetadata, StrategyContext,
    AssetClass, StrategyStyle, RiskCheckResult, OrderIntent,
)


class MockMomentumStrategy(BaseStrategy):
    """Mock momentum strategy for testing."""

    def get_metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            code="TEST_MOM_001",
            name="Test Momentum",
            source_book="151TS",
            asset_class=AssetClass.EQUITY,
            style=StrategyStyle.MOMENTUM,
            description="Test momentum strategy",
            parameters={"lookback": 252, "top_n": 50},
            parameter_bounds={"lookback": (20, 504), "top_n": (10, 200)},
        )

    def generate_features(self, context: StrategyContext, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        bars = data.get("bars_1d", pd.DataFrame())
        if bars.empty:
            return pd.DataFrame()
        # Simple momentum: 12-month return
        lookback = context.parameters.get("lookback", 252)
        features = bars.groupby("symbol")["close"].apply(
            lambda x: x.iloc[-1] / x.iloc[-min(lookback, len(x))] - 1
        ).to_frame("momentum")
        return features

    def generate_signal(self, features: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        if features.empty:
            return pd.Series(dtype=float)
        top_n = params.get("top_n", 50)
        signal = features["momentum"].rank(ascending=False)
        signal = signal.apply(lambda x: 1.0 if x <= top_n else -1.0 if x > len(signal) - top_n else 0.0)
        return signal


class TestStrategyMetadata:
    """Test strategy metadata."""

    def test_metadata_creation(self):
        strategy = MockMomentumStrategy()
        meta = strategy.get_metadata()
        assert meta.code == "TEST_MOM_001"
        assert meta.asset_class == AssetClass.EQUITY
        assert meta.style == StrategyStyle.MOMENTUM
        assert "lookback" in meta.parameters

    def test_parameter_validation_pass(self):
        strategy = MockMomentumStrategy()
        errors = strategy.validate_params({"lookback": 252, "top_n": 50})
        assert len(errors) == 0

    def test_parameter_validation_fail(self):
        strategy = MockMomentumStrategy()
        errors = strategy.validate_params({"lookback": 5, "top_n": 500})
        assert len(errors) == 2


class TestPositionSizing:
    """Test default position sizing."""

    def test_size_positions_basic(self):
        strategy = MockMomentumStrategy()
        signal = pd.Series({"AAPL": 1.0, "GOOGL": 0.5, "MSFT": -0.5, "TSLA": -1.0})
        params = {"book_size": 1_000_000, "max_single_weight": 0.05}
        targets = strategy.size_positions(signal, {}, params)
        assert abs(targets.sum()) < targets.abs().sum()  # Not all same direction
        assert targets.abs().max() <= 1_000_000 * 0.05 + 1  # Within weight limit

    def test_size_positions_zero_signal(self):
        strategy = MockMomentumStrategy()
        signal = pd.Series({"AAPL": 0.0, "GOOGL": 0.0})
        targets = strategy.size_positions(signal, {}, {"book_size": 1_000_000})
        assert (targets == 0).all()


class TestRiskCheck:
    """Test default risk checking."""

    def test_risk_check_pass(self):
        strategy = MockMomentumStrategy()
        targets = pd.Series({"AAPL": 50000, "GOOGL": 30000, "MSFT": -40000})
        result = strategy.check_risk(targets, {"limits": {"max_gross": 500000}})
        assert result.passed

    def test_risk_check_stale_data(self):
        strategy = MockMomentumStrategy()
        targets = pd.Series({"AAPL": 50000, "GOOGL": 30000})
        result = strategy.check_risk(targets, {
            "limits": {},
            "stale_symbols": ["AAPL"],
        })
        assert not result.passed
        assert len(result.hard_breaches) > 0


class TestOrderBuilding:
    """Test order building from position deltas."""

    def test_build_orders_new_positions(self):
        strategy = MockMomentumStrategy()
        current = pd.Series(dtype=float)
        targets = pd.Series({"AAPL": 100, "GOOGL": -50})
        orders = strategy.build_orders(current, targets)
        assert len(orders) == 2
        aapl_order = [o for o in orders if o.symbol == "AAPL"][0]
        assert aapl_order.side == "buy"
        assert aapl_order.qty == 100

    def test_build_orders_close_positions(self):
        strategy = MockMomentumStrategy()
        current = pd.Series({"AAPL": 100, "GOOGL": -50})
        targets = pd.Series(dtype=float)
        orders = strategy.build_orders(current, targets)
        assert len(orders) == 2

    def test_build_orders_no_change(self):
        strategy = MockMomentumStrategy()
        current = pd.Series({"AAPL": 100})
        targets = pd.Series({"AAPL": 100})
        orders = strategy.build_orders(current, targets)
        assert len(orders) == 0
