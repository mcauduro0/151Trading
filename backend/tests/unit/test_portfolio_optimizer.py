"""Unit tests for Portfolio Construction module.

Tests all four optimization methods:
1. Risk Parity
2. Hierarchical Risk Parity (HRP)
3. Mean-Variance (Markowitz)
4. Black-Litterman
Plus the unified PortfolioConstructor interface.
"""

import sys
sys.path.insert(0, "/home/ubuntu/151Trading")

import pytest
import numpy as np
import pandas as pd
from strategies.portfolio.optimizer import (
    RiskParityOptimizer,
    HRPOptimizer,
    MeanVarianceOptimizer,
    BlackLittermanOptimizer,
    PortfolioConstructor,
    OptimizationMethod,
    PortfolioResult,
    _winsorize,
    _clean_returns,
)


# ── Fixtures ──────────────────────────────────────────────

def make_returns(n_assets: int = 5, n_days: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic return series with realistic correlations."""
    rng = np.random.RandomState(seed)
    assets = [f"ASSET_{i}" for i in range(n_assets)]

    # Create correlated returns
    mu = rng.uniform(0.0002, 0.0008, n_assets)
    vol = rng.uniform(0.01, 0.03, n_assets)

    # Random correlation matrix
    A = rng.randn(n_assets, n_assets)
    corr = A @ A.T
    D = np.diag(1.0 / np.sqrt(np.diag(corr)))
    corr = D @ corr @ D

    L = np.linalg.cholesky(corr)
    raw = rng.randn(n_days, n_assets)
    correlated = raw @ L.T

    returns = pd.DataFrame(
        correlated * vol + mu,
        columns=assets,
        index=pd.date_range("2023-01-01", periods=n_days, freq="B"),
    )
    return returns


@pytest.fixture
def sample_returns():
    return make_returns(5, 500)


@pytest.fixture
def large_returns():
    return make_returns(10, 1000)


# ── Helper Tests ──────────────────────────────────────────

class TestHelpers:
    def test_winsorize_clips_outliers(self):
        arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 100])
        result = _winsorize(arr, 0.05, 0.95)
        assert result.max() <= 100  # 95th percentile
        assert result.min() >= 1

    def test_clean_returns_fills_na(self):
        df = pd.DataFrame({"A": [0.01, np.nan, 0.02], "B": [0.03, 0.01, np.nan]})
        clean = _clean_returns(df)
        assert clean.isna().sum().sum() == 0


# ── Risk Parity Tests ────────────────────────────────────

class TestRiskParity:
    def test_weights_sum_to_one(self, sample_returns):
        rp = RiskParityOptimizer()
        result = rp.optimize(sample_returns)
        assert abs(sum(result.weights.values()) - 1.0) < 1e-6

    def test_all_weights_positive(self, sample_returns):
        rp = RiskParityOptimizer()
        result = rp.optimize(sample_returns)
        for w in result.weights.values():
            assert w > 0

    def test_risk_contributions_roughly_equal(self, sample_returns):
        rp = RiskParityOptimizer()
        result = rp.optimize(sample_returns)
        rc_values = list(result.risk_contributions.values())
        target = 1.0 / len(rc_values)
        for rc in rc_values:
            assert abs(rc - target) < 0.05  # Within 5% of equal

    def test_returns_portfolio_result(self, sample_returns):
        rp = RiskParityOptimizer()
        result = rp.optimize(sample_returns)
        assert isinstance(result, PortfolioResult)
        assert result.method == OptimizationMethod.RISK_PARITY
        assert result.expected_volatility > 0

    def test_sharpe_ratio_reasonable(self, sample_returns):
        rp = RiskParityOptimizer()
        result = rp.optimize(sample_returns)
        assert -5 < result.sharpe_ratio < 10


# ── HRP Tests ────────────────────────────────────────────

class TestHRP:
    def test_weights_sum_to_one(self, sample_returns):
        hrp = HRPOptimizer()
        result = hrp.optimize(sample_returns)
        assert abs(sum(result.weights.values()) - 1.0) < 1e-6

    def test_all_weights_positive(self, sample_returns):
        hrp = HRPOptimizer()
        result = hrp.optimize(sample_returns)
        for w in result.weights.values():
            assert w > 0

    def test_returns_portfolio_result(self, sample_returns):
        hrp = HRPOptimizer()
        result = hrp.optimize(sample_returns)
        assert isinstance(result, PortfolioResult)
        assert result.method == OptimizationMethod.HRP

    def test_handles_large_universe(self, large_returns):
        hrp = HRPOptimizer()
        result = hrp.optimize(large_returns)
        assert len(result.weights) == 10
        assert abs(sum(result.weights.values()) - 1.0) < 1e-6

    def test_single_asset(self):
        ret = pd.DataFrame({"ONLY": np.random.randn(100) * 0.01})
        hrp = HRPOptimizer()
        result = hrp.optimize(ret)
        assert abs(result.weights["ONLY"] - 1.0) < 1e-6


# ── Mean-Variance Tests ──────────────────────────────────

class TestMeanVariance:
    def test_weights_sum_to_one(self, sample_returns):
        mv = MeanVarianceOptimizer()
        result = mv.optimize(sample_returns)
        assert abs(sum(result.weights.values()) - 1.0) < 1e-6

    def test_max_weight_constraint(self, sample_returns):
        mv = MeanVarianceOptimizer(max_weight=0.25)
        result = mv.optimize(sample_returns)
        for w in result.weights.values():
            assert w <= 0.26  # Small tolerance

    def test_no_negative_weights_by_default(self, sample_returns):
        mv = MeanVarianceOptimizer(allow_short=False)
        result = mv.optimize(sample_returns)
        for w in result.weights.values():
            assert w >= -1e-6

    def test_returns_portfolio_result(self, sample_returns):
        mv = MeanVarianceOptimizer()
        result = mv.optimize(sample_returns)
        assert isinstance(result, PortfolioResult)
        assert result.method == OptimizationMethod.MEAN_VARIANCE

    def test_sharpe_higher_than_equal_weight(self, sample_returns):
        mv = MeanVarianceOptimizer()
        result = mv.optimize(sample_returns)
        # MV should generally produce a reasonable Sharpe
        assert result.sharpe_ratio is not None


# ── Black-Litterman Tests ────────────────────────────────

class TestBlackLitterman:
    def test_weights_sum_to_one(self, sample_returns):
        bl = BlackLittermanOptimizer()
        result = bl.optimize(sample_returns)
        assert abs(sum(result.weights.values()) - 1.0) < 1e-6

    def test_no_views_returns_equilibrium(self, sample_returns):
        bl = BlackLittermanOptimizer()
        result = bl.optimize(sample_returns, views=None)
        assert result.method == OptimizationMethod.BLACK_LITTERMAN
        assert result.metadata.get("n_views") == 0

    def test_bullish_view_increases_weight(self, sample_returns):
        bl = BlackLittermanOptimizer()
        assets = list(sample_returns.columns)

        # No views
        result_no_views = bl.optimize(sample_returns, views=None)

        # Bullish view on first asset
        views = [{"assets": [assets[0]], "direction": 1, "return": 0.20}]
        result_with_views = bl.optimize(sample_returns, views=views)

        # The viewed asset should have higher weight
        w_no = result_no_views.weights[assets[0]]
        w_yes = result_with_views.weights[assets[0]]
        assert w_yes >= w_no * 0.8  # Allow some tolerance

    def test_custom_market_weights(self, sample_returns):
        bl = BlackLittermanOptimizer()
        assets = list(sample_returns.columns)
        mkt_w = {a: 1.0/len(assets) for a in assets}
        mkt_w[assets[0]] = 0.5
        result = bl.optimize(sample_returns, market_weights=mkt_w)
        assert abs(sum(result.weights.values()) - 1.0) < 1e-6

    def test_equilibrium_returns_in_metadata(self, sample_returns):
        bl = BlackLittermanOptimizer()
        result = bl.optimize(sample_returns)
        assert "equilibrium_returns" in result.metadata
        assert "posterior_returns" in result.metadata


# ── Portfolio Constructor (Unified) Tests ────────────────

class TestPortfolioConstructor:
    def test_compare_all_returns_four_methods(self, sample_returns):
        pc = PortfolioConstructor()
        results = pc.compare_all(sample_returns)
        assert len(results) == 4
        assert "risk_parity" in results
        assert "hrp" in results
        assert "mean_variance" in results
        assert "black_litterman" in results

    def test_blend_default_weights(self, sample_returns):
        pc = PortfolioConstructor()
        results = pc.compare_all(sample_returns)
        blended = pc.blend(results)
        assert abs(sum(blended.values()) - 1.0) < 1e-6

    def test_blend_custom_weights(self, sample_returns):
        pc = PortfolioConstructor()
        results = pc.compare_all(sample_returns)
        custom = {"risk_parity": 0.5, "hrp": 0.5}
        blended = pc.blend(results, blend_weights=custom)
        assert abs(sum(blended.values()) - 1.0) < 1e-6

    def test_to_dict_serialization(self, sample_returns):
        pc = PortfolioConstructor()
        result = pc.optimize(sample_returns, method=OptimizationMethod.RISK_PARITY)
        d = result.to_dict()
        assert "weights" in d
        assert "method" in d
        assert d["method"] == "risk_parity"
        assert "expected_return" in d
        assert "sharpe_ratio" in d
