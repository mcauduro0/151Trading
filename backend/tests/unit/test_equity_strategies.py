"""
Unit tests for Sprint 1 equity factor strategies.
Tests feature generation, signal generation, position sizing, and risk checks
using synthetic data to ensure deterministic, reproducible results.
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np
from datetime import date

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "strategies"))

from strategies.base import StrategyContext, AssetClass, StrategyStyle
from strategies.equity.momentum.cross_sectional import CrossSectionalMomentum
from strategies.equity.value.enhanced_value import EnhancedValueComposite
from strategies.equity.low_volatility.low_vol import LowVolatilityAnomaly
from strategies.equity.residual_momentum.residual_mom import ResidualMomentum


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_prices():
    """Generate 500 days of synthetic price data for 30 stocks."""
    np.random.seed(42)
    n_days = 500
    n_stocks = 30
    tickers = [f"STOCK_{i:02d}" for i in range(n_stocks)]
    dates = pd.bdate_range(end="2026-03-07", periods=n_days)

    # Generate prices with different drift rates (some trending up, some down)
    prices = pd.DataFrame(index=dates, columns=tickers, dtype=float)
    for i, ticker in enumerate(tickers):
        drift = 0.0003 * (i - n_stocks // 2)  # Range from -0.0045 to +0.0045
        vol = 0.01 + 0.005 * (i % 5)
        returns = np.random.normal(drift, vol, n_days)
        prices[ticker] = 100 * np.exp(np.cumsum(returns))

    return prices


@pytest.fixture
def synthetic_volumes(synthetic_prices):
    """Generate synthetic volume data."""
    np.random.seed(43)
    volumes = pd.DataFrame(
        np.random.uniform(1e6, 1e7, size=synthetic_prices.shape),
        index=synthetic_prices.index,
        columns=synthetic_prices.columns,
    )
    return volumes


@pytest.fixture
def synthetic_fundamentals(synthetic_prices):
    """Generate synthetic fundamental data."""
    np.random.seed(44)
    tickers = synthetic_prices.columns
    n = len(tickers)

    return pd.DataFrame({
        "bookValuePerShare": np.random.uniform(10, 100, n),
        "netIncomePerShare": np.random.uniform(1, 15, n),
        "operatingCashFlowPerShare": np.random.uniform(2, 20, n),
        "revenuePerShare": np.random.uniform(20, 200, n),
        "price": synthetic_prices.iloc[-1].values,
        "marketCap": np.random.uniform(5e9, 500e9, n),
        "sector": np.random.choice(
            ["Technology", "Healthcare", "Financials", "Consumer", "Energy"],
            n,
        ),
        "returnOnAssets": np.random.uniform(-0.05, 0.20, n),
        "returnOnAssets_prev": np.random.uniform(-0.05, 0.20, n),
        "currentRatio": np.random.uniform(0.5, 3.0, n),
        "currentRatio_prev": np.random.uniform(0.5, 3.0, n),
        "grossProfitMargin": np.random.uniform(0.1, 0.8, n),
        "grossProfitMargin_prev": np.random.uniform(0.1, 0.8, n),
        "longTermDebt": np.random.uniform(1e8, 1e10, n),
        "longTermDebt_prev": np.random.uniform(1e8, 1e10, n),
        "totalAssets": np.random.uniform(1e9, 1e11, n),
        "totalAssets_prev": np.random.uniform(1e9, 1e11, n),
        "weightedAverageShsOut": np.random.uniform(1e8, 1e9, n),
        "weightedAverageShsOut_prev": np.random.uniform(1e8, 1e9, n),
        "revenue": np.random.uniform(1e9, 1e11, n),
        "revenue_prev": np.random.uniform(1e9, 1e11, n),
    }, index=tickers)


@pytest.fixture
def benchmark_prices(synthetic_prices):
    """Generate synthetic benchmark (SPY) prices."""
    np.random.seed(45)
    n_days = len(synthetic_prices)
    returns = np.random.normal(0.0003, 0.01, n_days)
    spy = 400 * np.exp(np.cumsum(returns))
    return pd.Series(spy, index=synthetic_prices.index, name="SPY")


@pytest.fixture
def context(synthetic_prices):
    return StrategyContext(
        as_of_date=date(2026, 3, 7),
        universe=list(synthetic_prices.columns),
        parameters={},
    )


# ── Cross-Sectional Momentum Tests ───────────────────────────────────────

class TestCrossSectionalMomentum:

    def test_metadata(self):
        strat = CrossSectionalMomentum()
        meta = strat.get_metadata()
        assert meta.code == "EQ_MOM_001"
        assert meta.asset_class == AssetClass.EQUITY
        assert meta.style == StrategyStyle.MOMENTUM
        assert len(meta.assumptions) > 0
        assert len(meta.known_failure_modes) > 0

    def test_feature_generation(self, synthetic_prices, synthetic_volumes, context):
        strat = CrossSectionalMomentum()
        data = {"close": synthetic_prices, "volume": synthetic_volumes}
        features = strat.generate_features(context, data)

        assert "mom_12_1" in features.columns
        assert "mom_z_score" in features.columns
        assert len(features) > 0
        # Z-scores should be clipped to [-3, 3]
        assert features["mom_z_score"].max() <= 3.0
        assert features["mom_z_score"].min() >= -3.0

    def test_signal_generation(self, synthetic_prices, synthetic_volumes, context):
        strat = CrossSectionalMomentum()
        data = {"close": synthetic_prices, "volume": synthetic_volumes}
        features = strat.generate_features(context, data)
        signal = strat.generate_signal(features, strat._params)

        # Should have longs and shorts
        assert (signal > 0).sum() > 0
        assert (signal < 0).sum() > 0
        # Should be approximately market-neutral
        assert abs(signal.sum()) < 0.1

    def test_position_sizing(self, synthetic_prices, synthetic_volumes, context):
        strat = CrossSectionalMomentum()
        data = {"close": synthetic_prices, "volume": synthetic_volumes}
        features = strat.generate_features(context, data)
        signal = strat.generate_signal(features, strat._params)
        targets = strat.size_positions(signal, {}, strat._params)

        # Max single-name weight check
        book = strat._params["book_size"]
        max_weight = strat._params["max_single_weight"]
        assert targets.abs().max() <= book * max_weight * 1.01  # 1% tolerance

    def test_risk_check_vix_crash(self, synthetic_prices, synthetic_volumes, context):
        strat = CrossSectionalMomentum()
        data = {"close": synthetic_prices, "volume": synthetic_volumes}
        features = strat.generate_features(context, data)
        signal = strat.generate_signal(features, strat._params)
        targets = strat.size_positions(signal, {}, strat._params)

        # Normal VIX should pass
        result_normal = strat.check_risk(targets, {"vix_level": 20, "limits": {}})
        assert result_normal.passed

        # High VIX should fail
        result_crash = strat.check_risk(targets, {"vix_level": 50, "limits": {}})
        assert not result_crash.passed

    def test_parameter_validation(self):
        strat = CrossSectionalMomentum()
        errors = strat.validate_params({"lookback_days": 10})  # Below minimum
        assert len(errors) > 0

    def test_insufficient_data_raises(self, context):
        strat = CrossSectionalMomentum()
        short_prices = pd.DataFrame(
            np.random.randn(50, 10),
            columns=[f"S{i}" for i in range(10)],
        )
        with pytest.raises(ValueError, match="Need at least"):
            strat.generate_features(context, {"close": short_prices})


# ── Enhanced Value Composite Tests ────────────────────────────────────────

class TestEnhancedValueComposite:

    def test_metadata(self):
        strat = EnhancedValueComposite()
        meta = strat.get_metadata()
        assert meta.code == "EQ_VAL_002"
        assert meta.style == StrategyStyle.VALUE
        assert "F-Score" in meta.math_formula

    def test_feature_generation(self, synthetic_prices, synthetic_fundamentals, context):
        strat = EnhancedValueComposite()
        data = {"close": synthetic_prices, "fundamentals": synthetic_fundamentals}
        features = strat.generate_features(context, data)

        assert "book_to_market" in features.columns
        assert "earnings_yield" in features.columns
        assert "f_score" in features.columns
        # F-Score should be 0-9
        assert features["f_score"].max() <= 9
        assert features["f_score"].min() >= 0

    def test_fscore_computation(self, synthetic_fundamentals):
        strat = EnhancedValueComposite()
        fscore = strat._compute_fscore(synthetic_fundamentals)
        assert all(0 <= s <= 9 for s in fscore)

    def test_signal_with_quality_filter(self, synthetic_prices, synthetic_fundamentals, context):
        strat = EnhancedValueComposite({"min_market_cap": 0, "min_fscore": 5})
        data = {"close": synthetic_prices, "fundamentals": synthetic_fundamentals}
        features = strat.generate_features(context, data)
        signal = strat.generate_signal(features, strat._params)

        # Signal should exist (may be zero if no stocks pass quality filter)
        assert isinstance(signal, pd.Series)


# ── Low Volatility Anomaly Tests ─────────────────────────────────────────

class TestLowVolatilityAnomaly:

    def test_metadata(self):
        strat = LowVolatilityAnomaly()
        meta = strat.get_metadata()
        assert meta.code == "EQ_LVOL_003"
        assert meta.style == StrategyStyle.FACTOR

    def test_feature_generation(self, synthetic_prices, synthetic_volumes, benchmark_prices, context):
        strat = LowVolatilityAnomaly()
        data = {
            "close": synthetic_prices,
            "volume": synthetic_volumes,
            "benchmark": benchmark_prices,
        }
        features = strat.generate_features(context, data)

        assert "realized_vol_60d" in features.columns
        assert "beta_252d" in features.columns
        assert "low_vol_composite" in features.columns
        assert len(features) > 0

    def test_signal_inverse_vol_weighting(self, synthetic_prices, synthetic_volumes, benchmark_prices, context):
        strat = LowVolatilityAnomaly({"inverse_vol_weighting": True})
        data = {
            "close": synthetic_prices,
            "volume": synthetic_volumes,
            "benchmark": benchmark_prices,
        }
        features = strat.generate_features(context, data)
        signal = strat.generate_signal(features, strat._params)

        assert (signal > 0).sum() > 0  # Has longs
        assert (signal < 0).sum() > 0  # Has shorts

    def test_sector_limits(self, synthetic_prices, synthetic_volumes, benchmark_prices, context):
        strat = LowVolatilityAnomaly({"max_sector_weight": 0.25})
        sectors = pd.Series(
            np.random.choice(["Tech", "Health", "Finance"], len(synthetic_prices.columns)),
            index=synthetic_prices.columns,
        )
        data = {
            "close": synthetic_prices,
            "volume": synthetic_volumes,
            "benchmark": benchmark_prices,
            "sectors": sectors,
        }
        features = strat.generate_features(context, data)
        signal = strat.generate_signal(features, strat._params)
        assert isinstance(signal, pd.Series)


# ── Residual Momentum Tests ──────────────────────────────────────────────

class TestResidualMomentum:

    def test_metadata(self):
        strat = ResidualMomentum()
        meta = strat.get_metadata()
        assert meta.code == "EQ_RMOM_004"
        assert meta.style == StrategyStyle.MOMENTUM
        assert "Fama-French" in meta.description

    def test_feature_generation(self, synthetic_prices, synthetic_volumes, context):
        strat = ResidualMomentum({"min_observations": 100, "min_r_squared": 0.0})
        data = {"close": synthetic_prices, "volume": synthetic_volumes}
        features = strat.generate_features(context, data)

        if len(features) > 0:
            assert "residual_mom" in features.columns
            assert "alpha" in features.columns
            assert "beta_mkt" in features.columns
            assert "r_squared" in features.columns

    def test_signal_generation(self, synthetic_prices, synthetic_volumes, context):
        strat = ResidualMomentum({"min_observations": 100, "min_r_squared": 0.0})
        data = {"close": synthetic_prices, "volume": synthetic_volumes}
        features = strat.generate_features(context, data)

        if len(features) >= 20:
            signal = strat.generate_signal(features, strat._params)
            assert (signal > 0).sum() > 0
            assert (signal < 0).sum() > 0

    def test_approximate_factors(self, synthetic_prices, context):
        strat = ResidualMomentum()
        returns = synthetic_prices.pct_change().dropna()
        mkt_rf, smb, hml, rf = strat._construct_approximate_factors(returns, {})
        assert len(mkt_rf) > 0
        assert isinstance(mkt_rf, pd.Series)


# ── Cross-Strategy Tests ─────────────────────────────────────────────────

class TestCrossStrategy:

    def test_all_strategies_implement_interface(self):
        """All strategies must implement the full BaseStrategy interface."""
        strategies = [
            CrossSectionalMomentum(),
            EnhancedValueComposite(),
            LowVolatilityAnomaly(),
            ResidualMomentum(),
        ]
        for strat in strategies:
            meta = strat.get_metadata()
            assert meta.code is not None
            assert meta.name is not None
            assert meta.asset_class is not None
            assert meta.style is not None
            assert len(meta.required_data) > 0

    def test_all_strategies_have_unique_codes(self):
        strategies = [
            CrossSectionalMomentum(),
            EnhancedValueComposite(),
            LowVolatilityAnomaly(),
            ResidualMomentum(),
        ]
        codes = [s.get_metadata().code for s in strategies]
        assert len(codes) == len(set(codes))

    def test_parameter_bounds_are_valid(self):
        strategies = [
            CrossSectionalMomentum(),
            EnhancedValueComposite(),
            LowVolatilityAnomaly(),
            ResidualMomentum(),
        ]
        for strat in strategies:
            meta = strat.get_metadata()
            for param, (lower, upper) in meta.parameter_bounds.items():
                assert lower < upper, f"{meta.code}: {param} bounds invalid"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
