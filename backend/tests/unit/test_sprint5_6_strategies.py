"""
Unit tests for Sprint 5 (Fixed Income) and Sprint 6 (FX & Rates) strategies.
Tests cover: Yield Curve, Duration Timing, Butterfly Spread, G10 Carry, PPP Value, Triangular Arb.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import pytest
import numpy as np

# Fixed Income strategies
from strategies.fixed_income.yield_curve import (
    YieldCurveStrategy, CurveRegime, FRED_YIELD_SERIES
)
from strategies.fixed_income.duration_timing import (
    DurationTimingStrategy, DurationFactors
)
from strategies.fixed_income.butterfly_spread import (
    ButterflySpreadStrategy, BUTTERFLY_CONFIGS
)

# FX strategies
from strategies.fx.g10_carry import (
    G10CarryStrategy, CurrencyCarryData
)
from strategies.base import SignalDirection
from strategies.fx.ppp_value import (
    PPPValueStrategy, PPPCurrencyData
)
from strategies.fx.triangular_arb import (
    TriangularArbStrategy, FXQuote
)


# ============================================================
# Yield Curve Strategy Tests
# ============================================================

class TestYieldCurveStrategy:
    
    def setup_method(self):
        self.strategy = YieldCurveStrategy()
    
    def test_build_normal_curve(self):
        """Test building a normal (upward-sloping) yield curve."""
        yields = {
            "3M": 4.5, "6M": 4.4, "1Y": 4.3, "2Y": 4.2,
            "3Y": 4.1, "5Y": 4.0, "7Y": 4.1, "10Y": 4.3,
            "20Y": 4.5, "30Y": 4.6,
        }
        curve = self.strategy.build_curve(yields)
        
        assert curve.slope_2s10s == pytest.approx(0.1, abs=0.01)
        assert curve.slope_3m10y == pytest.approx(-0.2, abs=0.01)
        assert len(curve.points) == 10
    
    def test_inverted_curve_detection(self):
        """Test detection of inverted yield curve."""
        yields = {
            "3M": 5.5, "2Y": 5.0, "5Y": 4.5, "10Y": 4.2, "30Y": 4.3,
        }
        curve = self.strategy.build_curve(yields)
        
        assert curve.slope_2s10s < 0
        assert curve.regime in (CurveRegime.INVERTED, CurveRegime.DEEPLY_INVERTED)
    
    def test_steep_curve_regime(self):
        """Test steep curve regime classification."""
        yields = {"3M": 2.0, "2Y": 2.5, "5Y": 3.5, "10Y": 4.5, "30Y": 5.0}
        curve = self.strategy.build_curve(yields)
        
        assert curve.slope_2s10s == pytest.approx(2.0, abs=0.01)
        assert curve.regime == CurveRegime.NORMAL_STEEP
    
    def test_spread_zscore_computation(self):
        """Test z-score computation for spread history."""
        history = list(np.random.normal(1.0, 0.3, 252))
        z = self.strategy.compute_spread_zscore(2.0, history)
        
        assert z > 2.0  # 2.0 is well above mean of 1.0
    
    def test_flattener_signal_generation(self):
        """Test that wide 2s10s spread generates flattener signal."""
        yields = {"3M": 3.0, "2Y": 3.5, "5Y": 4.0, "10Y": 5.5, "30Y": 6.0}
        curve = self.strategy.build_curve(yields)
        
        # History with mean ~1.0, current spread is 2.0
        history = list(np.random.normal(1.0, 0.3, 252))
        
        signals = self.strategy.generate_signals(
            curve, spread_history_2s10s=history
        )
        
        flatteners = [s for s in signals if "FLATTENER" in s.symbol]
        assert len(flatteners) >= 1
    
    def test_inversion_warning_signal(self):
        """Test that inverted curve generates warning signal."""
        yields = {"3M": 5.5, "2Y": 5.0, "5Y": 4.5, "10Y": 4.2, "30Y": 4.3}
        curve = self.strategy.build_curve(yields)
        
        signals = self.strategy.generate_signals(curve)
        
        warnings = [s for s in signals if "INVERSION" in s.symbol]
        assert len(warnings) == 1
    
    def test_butterfly_curvature(self):
        """Test butterfly curvature computation."""
        yields = {"2Y": 4.0, "5Y": 4.5, "10Y": 4.2, "30Y": 4.6}
        curve = self.strategy.build_curve(yields)
        
        # 2s5s10s = 2*4.5 - 4.0 - 4.2 = 0.8
        assert curve.curvature_2s5s10s == pytest.approx(0.8, abs=0.01)


# ============================================================
# Duration Timing Strategy Tests
# ============================================================

class TestDurationTimingStrategy:
    
    def setup_method(self):
        self.strategy = DurationTimingStrategy()
    
    def test_steep_curve_long_duration(self):
        """Steep curve with low inflation should favor long duration."""
        factors = DurationFactors(
            yield_10y=4.5, yield_2y=3.0, slope_2s10s=1.5,
            vix=15, tlt_momentum_20d=0.02, tlt_momentum_60d=0.04,
            cpi_yoy=2.0, pmi=52, hy_spread=350,
        )
        score = self.strategy.compute_duration_score(factors)
        assert score > 0.2  # Should favor longer duration
    
    def test_crisis_flight_to_quality(self):
        """High VIX + wide spreads should push to long duration."""
        factors = DurationFactors(
            yield_10y=3.5, yield_2y=4.0, slope_2s10s=-0.5,
            vix=40, tlt_momentum_20d=0.05, tlt_momentum_60d=0.08,
            cpi_yoy=3.0, pmi=45, hy_spread=700, hy_spread_change_1m=200,
        )
        score = self.strategy.compute_duration_score(factors)
        assert score > 0.3  # Crisis → flight to quality → long duration
    
    def test_high_inflation_short_duration(self):
        """High inflation should favor short duration."""
        factors = DurationFactors(
            yield_10y=5.0, yield_2y=5.5, slope_2s10s=-0.5,
            vix=18, tlt_momentum_20d=-0.03, tlt_momentum_60d=-0.06,
            cpi_yoy=5.0, pmi=55, hy_spread=300,
        )
        score = self.strategy.compute_duration_score(factors)
        assert score < 0  # High inflation → short duration
    
    def test_allocation_mapping(self):
        """Test score to ETF allocation mapping."""
        alloc_long = self.strategy.score_to_allocation(0.7)
        assert alloc_long["TLT"] > 0.5
        
        alloc_short = self.strategy.score_to_allocation(-0.7)
        assert alloc_short["BIL"] > 0.5
    
    def test_signal_generation(self):
        """Test that signals are generated with proper ETF symbols."""
        factors = DurationFactors(
            yield_10y=4.0, yield_2y=3.5, slope_2s10s=0.5,
            vix=20, cpi_yoy=2.5, pmi=50,
        )
        signals = self.strategy.generate_signals(factors)
        
        assert len(signals) > 0
        symbols = {s.symbol for s in signals}
        assert symbols.issubset({"TLT", "IEF", "SHY", "BIL"})


# ============================================================
# Butterfly Spread Strategy Tests
# ============================================================

class TestButterflySpreadStrategy:
    
    def setup_method(self):
        self.strategy = ButterflySpreadStrategy()
    
    def test_curvature_computation(self):
        """Test butterfly curvature calculation."""
        yields = {"2Y": 4.0, "5Y": 4.5, "10Y": 4.2}
        bfly = BUTTERFLY_CONFIGS["2s5s10s"]
        
        curvature = self.strategy.compute_curvature(yields, bfly)
        # 2*4.5 - 4.0 - 4.2 = 0.8
        assert curvature == pytest.approx(0.8, abs=0.01)
    
    def test_carry_computation(self):
        """Test carry estimation for butterfly."""
        yields = {"2Y": 4.0, "5Y": 4.5, "10Y": 4.2}
        bfly = BUTTERFLY_CONFIGS["2s5s10s"]
        
        carry = self.strategy.compute_carry(yields, bfly)
        assert carry > 0  # Belly (5Y=4.5) yields more than wings
    
    def test_sell_belly_signal(self):
        """High curvature should generate sell-belly signal."""
        yields = {"2Y": 4.0, "5Y": 5.0, "10Y": 4.2, "30Y": 4.5}
        
        # History with mean ~0.2, current curvature is much higher
        history_2s5s10s = list(np.random.normal(0.2, 0.1, 252))
        
        signals = self.strategy.generate_signals(
            yields, curvature_histories={"2s5s10s": history_2s5s10s}
        )
        
        sell_belly = [s for s in signals if "SELL_BELLY" in s.symbol]
        assert len(sell_belly) >= 1
    
    def test_missing_tenor_returns_none(self):
        """Missing tenor should return None curvature."""
        yields = {"2Y": 4.0, "10Y": 4.2}  # Missing 5Y
        bfly = BUTTERFLY_CONFIGS["2s5s10s"]
        
        curvature = self.strategy.compute_curvature(yields, bfly)
        assert curvature is None


# ============================================================
# G10 Carry Trade Strategy Tests
# ============================================================

class TestG10CarryStrategy:
    
    def setup_method(self):
        self.strategy = G10CarryStrategy()
    
    def _make_currencies(self):
        """Create sample G10 currency data."""
        return [
            CurrencyCarryData(currency="AUD", short_rate=4.5, fx_return_3m=0.02, fx_vol_3m=0.10),
            CurrencyCarryData(currency="NZD", short_rate=5.0, fx_return_3m=0.01, fx_vol_3m=0.11),
            CurrencyCarryData(currency="NOK", short_rate=4.0, fx_return_3m=-0.01, fx_vol_3m=0.12),
            CurrencyCarryData(currency="GBP", short_rate=4.5, fx_return_3m=0.03, fx_vol_3m=0.08),
            CurrencyCarryData(currency="EUR", short_rate=3.5, fx_return_3m=0.01, fx_vol_3m=0.07),
            CurrencyCarryData(currency="CAD", short_rate=4.0, fx_return_3m=0.00, fx_vol_3m=0.08),
            CurrencyCarryData(currency="SEK", short_rate=3.0, fx_return_3m=-0.02, fx_vol_3m=0.10),
            CurrencyCarryData(currency="CHF", short_rate=1.5, fx_return_3m=0.02, fx_vol_3m=0.06),
            CurrencyCarryData(currency="JPY", short_rate=0.1, fx_return_3m=-0.03, fx_vol_3m=0.09),
        ]
    
    def test_carry_ranking(self):
        """High-yield currencies should be ranked for long positions."""
        currencies = self._make_currencies()
        signals = self.strategy.generate_signals(currencies, usd_rate=5.0, vix=18)
        
        assert len(signals) == 6  # 3 long + 3 short
        
        longs = [s for s in signals if s.direction == SignalDirection.LONG]
        shorts = [s for s in signals if s.direction == SignalDirection.SHORT]
        
        assert len(longs) == 3
        assert len(shorts) == 3
    
    def test_vix_crash_protection(self):
        """High VIX should reduce position sizes."""
        currencies = self._make_currencies()
        
        signals_normal = self.strategy.generate_signals(currencies, usd_rate=5.0, vix=15)
        signals_crisis = self.strategy.generate_signals(currencies, usd_rate=5.0, vix=40)
        
        # Crisis weights should be smaller
        max_w_normal = max(s.weight for s in signals_normal)
        max_w_crisis = max(s.weight for s in signals_crisis)
        assert max_w_crisis < max_w_normal
    
    def test_jpy_likely_shorted(self):
        """JPY with 0.1% rate should be in short basket."""
        currencies = self._make_currencies()
        signals = self.strategy.generate_signals(currencies, usd_rate=5.0, vix=18)
        
        jpy_signals = [s for s in signals if "JPY" in s.symbol]
        assert len(jpy_signals) > 0
        assert jpy_signals[0].direction == SignalDirection.SHORT


# ============================================================
# PPP Value Strategy Tests
# ============================================================

class TestPPPValueStrategy:
    
    def setup_method(self):
        self.strategy = PPPValueStrategy()
    
    def test_undervalued_currency_long(self):
        """Significantly undervalued currency should generate long signal."""
        currencies = [
            PPPCurrencyData(
                currency="GBP", spot_rate=1.10, ppp_fair_value=1.50,
                reer_index=90, reer_5y_avg=100, fx_return_6m=0.02,
            ),
            PPPCurrencyData(
                currency="EUR", spot_rate=1.08, ppp_fair_value=1.10,
                reer_index=98, reer_5y_avg=100, fx_return_6m=0.01,
            ),
            PPPCurrencyData(
                currency="CHF", spot_rate=1.20, ppp_fair_value=0.90,
                reer_index=115, reer_5y_avg=100, fx_return_6m=0.03,
            ),
        ]
        
        signals = self.strategy.generate_signals(currencies)
        
        gbp_long = [s for s in signals if "GBP" in s.symbol and s.direction == SignalDirection.LONG]
        assert len(gbp_long) >= 1  # GBP is 27% undervalued
    
    def test_overvalued_currency_short(self):
        """Significantly overvalued currency should generate short signal."""
        currencies = [
            PPPCurrencyData(
                currency="CHF", spot_rate=1.30, ppp_fair_value=0.95,
                reer_index=120, reer_5y_avg=100, fx_return_6m=0.05,
            ),
            PPPCurrencyData(
                currency="JPY", spot_rate=0.0067, ppp_fair_value=0.0090,
                reer_index=80, reer_5y_avg=100, fx_return_6m=-0.05,
            ),
        ]
        
        signals = self.strategy.generate_signals(currencies)
        
        chf_short = [s for s in signals if "CHF" in s.symbol and s.direction == SignalDirection.SHORT]
        assert len(chf_short) >= 1  # CHF is ~37% overvalued
    
    def test_small_misalignment_no_signal(self):
        """Small misalignment should not generate signals."""
        currencies = [
            PPPCurrencyData(
                currency="EUR", spot_rate=1.08, ppp_fair_value=1.10,
                reer_index=99, reer_5y_avg=100,
            ),
            PPPCurrencyData(
                currency="GBP", spot_rate=1.28, ppp_fair_value=1.30,
                reer_index=100, reer_5y_avg=100,
            ),
        ]
        
        signals = self.strategy.generate_signals(currencies)
        assert len(signals) == 0  # Both within 2% of fair value


# ============================================================
# Triangular Arbitrage Strategy Tests
# ============================================================

class TestTriangularArbStrategy:
    
    def setup_method(self):
        self.strategy = TriangularArbStrategy({"min_profit_bps": 0.5})
    
    def _make_consistent_quotes(self):
        """Create consistent (no-arb) FX quotes."""
        return {
            "EURUSD": FXQuote(pair="EURUSD", base="EUR", quote="USD", bid=1.0795, ask=1.0805),
            "USDJPY": FXQuote(pair="USDJPY", base="USD", quote="JPY", bid=149.95, ask=150.05),
            "EURJPY": FXQuote(pair="EURJPY", base="EUR", quote="JPY", bid=161.90, ask=162.10),
        }
    
    def _make_mispriced_quotes(self):
        """Create mispriced FX quotes with arbitrage opportunity."""
        return {
            "EURUSD": FXQuote(pair="EURUSD", base="EUR", quote="USD", bid=1.0800, ask=1.0802),
            "USDJPY": FXQuote(pair="USDJPY", base="USD", quote="JPY", bid=150.00, ask=150.02),
            "EURJPY": FXQuote(pair="EURJPY", base="EUR", quote="JPY", bid=160.00, ask=160.05),
            # Implied EURJPY = 1.0801 * 150.01 = 162.02, but market is 160.025
            # Deviation ≈ 124 bps
        }
    
    def test_find_triangles(self):
        """Test triangle detection from available quotes."""
        quotes = self._make_consistent_quotes()
        triangles = self.strategy.find_triangles(quotes)
        
        assert len(triangles) >= 1
    
    def test_consistent_quotes_small_deviation(self):
        """Consistent quotes should have near-zero deviation."""
        quotes = self._make_consistent_quotes()
        opportunities = self.strategy.scan_opportunities(quotes)
        
        # Should find the triangle but deviation should be small
        if opportunities:
            assert abs(opportunities[0].deviation_bps) < 50  # Less than 50 bps
    
    def test_mispriced_quotes_detected(self):
        """Mispriced quotes should be detected as opportunity."""
        quotes = self._make_mispriced_quotes()
        opportunities = self.strategy.scan_opportunities(quotes)
        
        assert len(opportunities) >= 1
        assert abs(opportunities[0].deviation_bps) > 10  # Significant deviation
    
    def test_signal_generation_with_opportunity(self):
        """Mispriced quotes should generate trading signals."""
        quotes = self._make_mispriced_quotes()
        signals = self.strategy.generate_signals(quotes)
        
        assert len(signals) >= 1
        assert any("TRIANGLE" in s.symbol for s in signals)


# ============================================================
# Cross-Strategy Integration Tests
# ============================================================

class TestCrossStrategyIntegration:
    
    def test_all_strategies_have_required_properties(self):
        """All strategies must implement required interface."""
        strategies = [
            YieldCurveStrategy(),
            DurationTimingStrategy(),
            ButterflySpreadStrategy(),
            G10CarryStrategy(),
            PPPValueStrategy(),
            TriangularArbStrategy(),
        ]
        
        for s in strategies:
            assert hasattr(s, 'strategy_id')
            assert hasattr(s, 'name')
            assert hasattr(s, 'asset_class')
            assert hasattr(s, 'description')
            assert s.strategy_id is not None
            assert len(s.name) > 0
    
    def test_all_strategies_return_list_on_empty_data(self):
        """All strategies should return empty list on empty data."""
        strategies_and_data = [
            (YieldCurveStrategy(), {"yields": {}}),
            (DurationTimingStrategy(), {}),
            (ButterflySpreadStrategy(), {"yields": {}}),
            (G10CarryStrategy(), {"currencies": [], "usd_rate": 5.0}),
            (PPPValueStrategy(), {"currencies": []}),
            (TriangularArbStrategy(), {"quotes": {}}),
        ]
        
        for strategy, data in strategies_and_data:
            result = strategy.run(data)
            assert isinstance(result, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
