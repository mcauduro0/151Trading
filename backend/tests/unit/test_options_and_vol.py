"""
Unit tests for Sprint 3 (Options Engine) and Sprint 4 (Volatility Strategies).
Tests: pricing, Greeks, 58 structures, payoff analysis, and 4 vol strategies.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


# ============================================================
# OPTIONS ENGINE TESTS
# ============================================================

class TestBlackScholesPricing:
    """Test Black-Scholes pricing engine."""

    def test_call_price_atm(self):
        from strategies.options.engine.pricing import bs_price, OptionType
        price = bs_price(S=100, K=100, T=30/365, r=0.05, sigma=0.20, option_type=OptionType.CALL)
        assert 1.0 < price < 5.0, f"ATM call should be 1-5, got {price}"

    def test_put_price_atm(self):
        from strategies.options.engine.pricing import bs_price, OptionType
        price = bs_price(S=100, K=100, T=30/365, r=0.05, sigma=0.20, option_type=OptionType.PUT)
        assert 1.0 < price < 5.0, f"ATM put should be 1-5, got {price}"

    def test_put_call_parity(self):
        """C - P = S - K*exp(-rT)"""
        from strategies.options.engine.pricing import bs_price, OptionType
        import math
        S, K, T, r, sigma = 100, 100, 0.25, 0.05, 0.20
        call = bs_price(S, K, T, r, sigma, OptionType.CALL)
        put = bs_price(S, K, T, r, sigma, OptionType.PUT)
        parity = S - K * math.exp(-r * T)
        assert abs((call - put) - parity) < 0.01, "Put-call parity violated"

    def test_deep_itm_call(self):
        from strategies.options.engine.pricing import bs_price, OptionType
        price = bs_price(S=150, K=100, T=30/365, r=0.05, sigma=0.20, option_type=OptionType.CALL)
        assert price > 49, f"Deep ITM call should be > 49, got {price}"

    def test_deep_otm_call(self):
        from strategies.options.engine.pricing import bs_price, OptionType
        price = bs_price(S=50, K=100, T=30/365, r=0.05, sigma=0.20, option_type=OptionType.CALL)
        assert price < 0.01, f"Deep OTM call should be ~0, got {price}"

    def test_expired_option(self):
        from strategies.options.engine.pricing import bs_price, OptionType
        call = bs_price(S=110, K=100, T=0, r=0.05, sigma=0.20, option_type=OptionType.CALL)
        assert abs(call - 10) < 0.01, f"Expired ITM call should be intrinsic=10, got {call}"
        put = bs_price(S=110, K=100, T=0, r=0.05, sigma=0.20, option_type=OptionType.PUT)
        assert abs(put) < 0.01, f"Expired OTM put should be 0, got {put}"


class TestGreeks:
    """Test Greeks computation."""

    def test_call_delta_range(self):
        from strategies.options.engine.pricing import bs_greeks, OptionType
        g = bs_greeks(S=100, K=100, T=30/365, r=0.05, sigma=0.20, option_type=OptionType.CALL)
        assert 0.4 < g.delta < 0.6, f"ATM call delta should be ~0.5, got {g.delta}"

    def test_put_delta_range(self):
        from strategies.options.engine.pricing import bs_greeks, OptionType
        g = bs_greeks(S=100, K=100, T=30/365, r=0.05, sigma=0.20, option_type=OptionType.PUT)
        assert -0.6 < g.delta < -0.4, f"ATM put delta should be ~-0.5, got {g.delta}"

    def test_gamma_positive(self):
        from strategies.options.engine.pricing import bs_greeks, OptionType
        g = bs_greeks(S=100, K=100, T=30/365, r=0.05, sigma=0.20, option_type=OptionType.CALL)
        assert g.gamma > 0, f"Gamma should be positive, got {g.gamma}"

    def test_theta_negative_for_long(self):
        from strategies.options.engine.pricing import bs_greeks, OptionType
        g = bs_greeks(S=100, K=100, T=30/365, r=0.05, sigma=0.20, option_type=OptionType.CALL)
        assert g.theta < 0, f"Long call theta should be negative, got {g.theta}"

    def test_vega_positive(self):
        from strategies.options.engine.pricing import bs_greeks, OptionType
        g = bs_greeks(S=100, K=100, T=30/365, r=0.05, sigma=0.20, option_type=OptionType.CALL)
        assert g.vega > 0, f"Vega should be positive, got {g.vega}"


class TestImpliedVolatility:
    """Test IV solver."""

    def test_iv_roundtrip(self):
        from strategies.options.engine.pricing import bs_price, implied_volatility, OptionType
        S, K, T, r, true_iv = 100, 100, 0.25, 0.05, 0.25
        price = bs_price(S, K, T, r, true_iv, OptionType.CALL)
        solved_iv = implied_volatility(price, S, K, T, r, OptionType.CALL)
        assert abs(solved_iv - true_iv) < 0.001, f"IV roundtrip failed: {solved_iv} vs {true_iv}"

    def test_iv_put(self):
        from strategies.options.engine.pricing import bs_price, implied_volatility, OptionType
        S, K, T, r, true_iv = 100, 95, 0.5, 0.05, 0.30
        price = bs_price(S, K, T, r, true_iv, OptionType.PUT)
        solved_iv = implied_volatility(price, S, K, T, r, OptionType.PUT)
        assert abs(solved_iv - true_iv) < 0.001


class TestPayoffEngine:
    """Test payoff calculations."""

    def test_long_call_payoff(self):
        from strategies.options.engine.pricing import (
            OptionLeg, OptionType, PositionSide, leg_payoff_at_expiry
        )
        leg = OptionLeg(OptionType.CALL, 100, 30, PositionSide.LONG, 1, 3.0)
        spots = np.array([90, 100, 103, 110, 120])
        pnl = leg_payoff_at_expiry(leg, spots)
        expected = np.array([-3, -3, 0, 7, 17])
        np.testing.assert_array_almost_equal(pnl, expected)

    def test_iron_condor_defined_risk(self):
        from strategies.options.engine.pricing import (
            OptionLeg, OptionType, PositionSide, structure_payoff_at_expiry,
            max_profit, max_loss
        )
        legs = [
            OptionLeg(OptionType.PUT, 90, 30, PositionSide.LONG, 1, 0.50),
            OptionLeg(OptionType.PUT, 95, 30, PositionSide.SHORT, 1, 1.50),
            OptionLeg(OptionType.CALL, 105, 30, PositionSide.SHORT, 1, 1.50),
            OptionLeg(OptionType.CALL, 110, 30, PositionSide.LONG, 1, 0.50),
        ]
        spots = np.linspace(80, 120, 500)
        mp = max_profit(legs, spots)
        ml = max_loss(legs, spots)
        assert mp > 0, "Iron condor should have positive max profit"
        assert ml < 0, "Iron condor should have negative max loss"
        assert abs(ml) < 10, "Iron condor max loss should be bounded"

    def test_breakeven_points(self):
        from strategies.options.engine.pricing import (
            OptionLeg, OptionType, PositionSide, breakeven_points
        )
        # Long straddle: 2 breakevens
        legs = [
            OptionLeg(OptionType.CALL, 100, 30, PositionSide.LONG, 1, 3.0),
            OptionLeg(OptionType.PUT, 100, 30, PositionSide.LONG, 1, 3.0),
        ]
        spots = np.linspace(80, 120, 1000)
        be = breakeven_points(legs, spots)
        assert len(be) == 2, f"Straddle should have 2 breakevens, got {len(be)}"
        assert abs(be[0] - 94) < 1, f"Lower breakeven should be ~94, got {be[0]}"
        assert abs(be[1] - 106) < 1, f"Upper breakeven should be ~106, got {be[1]}"


class TestStructureRegistry:
    """Test the 58 structure registry."""

    def test_total_structure_count(self):
        from strategies.options.structures.families import count_structures
        total = count_structures()
        assert total == 58, f"Expected 58 structures, got {total}"

    def test_all_families_present(self):
        from strategies.options.structures.families import get_all_families
        families = get_all_families()
        expected_families = [
            "Single Legs", "Vertical Spreads", "Butterflies & Condors",
            "Straddles & Strangles", "Calendar & Diagonal Spreads",
            "Ratio Spreads", "Synthetic & Conversion", "Complex / Exotic Combos",
        ]
        for fam in expected_families:
            assert fam in families, f"Missing family: {fam}"

    def test_family_counts(self):
        from strategies.options.structures.families import get_all_families
        families = get_all_families()
        expected = {
            "Single Legs": 4,
            "Vertical Spreads": 8,
            "Butterflies & Condors": 10,
            "Straddles & Strangles": 8,
            "Calendar & Diagonal Spreads": 8,
            "Ratio Spreads": 8,
            "Synthetic & Conversion": 6,
            "Complex / Exotic Combos": 6,
        }
        for fam, count in expected.items():
            actual = len(families[fam])
            assert actual == count, f"{fam}: expected {count}, got {actual}"

    def test_each_structure_builds(self):
        """Verify every structure can be instantiated."""
        from strategies.options.structures.families import list_structures
        S = 100
        for sd in list_structures():
            fn = sd.builder
            try:
                if sd.family == "Single Legs":
                    legs = fn(S, 100, 30, 3.0)
                elif sd.family == "Vertical Spreads":
                    legs = fn(S, 95, 105, 30, 3.0, 1.5)
                elif sd.family == "Butterflies & Condors":
                    if sd.max_legs == 3:
                        legs = fn(S, 90, 100, 110, 30, 1.0, 3.0, 1.0)
                    elif sd.id == "BC_04":  # Iron butterfly
                        legs = fn(S, 90, 100, 110, 30, 0.5, 3.0, 3.0, 0.5)
                    else:
                        legs = fn(S, 90, 95, 105, 110, 30, 0.5, 1.5, 1.5, 0.5)
                elif sd.family == "Straddles & Strangles":
                    if "Guts" in sd.name:
                        legs = fn(S, 95, 105, 30, 3.0, 3.0)
                    elif "Straddle" in sd.name or "Strap" in sd.name or "Strip" in sd.name:
                        legs = fn(S, 100, 30, 3.0, 3.0)
                    else:
                        legs = fn(S, 95, 105, 30, 2.0, 2.0)
                elif sd.family == "Calendar & Diagonal Spreads":
                    if "Diagonal" in sd.name:
                        legs = fn(S, 95, 105, 15, 45, 2.0, 3.0)
                    else:
                        legs = fn(S, 100, 15, 45, 2.0, 3.0)
                elif sd.family == "Ratio Spreads":
                    if "Christmas" in sd.name:
                        legs = fn(S, 100, 105, 110, 30, 3.0, 1.5, 0.5)
                    elif "Jade" in sd.name:
                        legs = fn(S, 90, 105, 110, 30, 1.5, 2.0, 0.5)
                    elif "Twisted" in sd.name:
                        legs = fn(S, 90, 95, 110, 30, 0.5, 1.5, 2.0)
                    else:
                        legs = fn(S, 95, 105, 30, 3.0, 1.5)
                elif sd.family == "Synthetic & Conversion":
                    if "Collar" in sd.name or "Risk Reversal" in sd.name:
                        legs = fn(S, 95, 105, 30, 2.0, 2.0)
                    else:
                        legs = fn(S, 100, 30, 3.0, 3.0)
                elif sd.family == "Complex / Exotic Combos":
                    if "Double Diagonal" in sd.name:
                        legs = fn(S, 95, 105, 95, 105, 15, 45, 1.5, 1.5, 2.0, 2.0)
                    elif "Double Calendar" in sd.name:
                        legs = fn(S, 95, 105, 15, 45, 1.5, 1.5, 2.0, 2.0)
                    elif "Seagull" in sd.name:
                        legs = fn(S, 90, 100, 110, 30, 1.0, 3.0, 1.0)
                    elif "ZEBRA" in sd.name:
                        legs = fn(S, 95, 100, 30, 4.0, 3.0)
                    elif "Big Lizard" in sd.name:
                        legs = fn(S, 100, 110, 30, 3.0, 3.0)
                    else:
                        continue
                else:
                    continue

                assert isinstance(legs, list), f"{sd.name}: should return list"
                assert len(legs) > 0, f"{sd.name}: should have legs"
            except Exception as e:
                pytest.fail(f"Structure {sd.id} ({sd.name}) failed to build: {e}")


# ============================================================
# VOLATILITY STRATEGY TESTS
# ============================================================

def _make_vix_df(values, col="value"):
    dates = pd.date_range(end=datetime.now(), periods=len(values), freq="B")
    return pd.DataFrame({col: values}, index=dates)

def _make_spy_df(start=400, n=252):
    dates = pd.date_range(end=datetime.now(), periods=n, freq="B")
    np.random.seed(42)
    returns = np.random.normal(0.0003, 0.01, n)
    prices = start * np.cumprod(1 + returns)
    return pd.DataFrame({"close": prices}, index=dates)


class TestVIXBasisTrade:
    """Test VIX Basis Trade strategy."""

    def test_contango_signal(self):
        from strategies.volatility.vix_basis import VIXBasisTrade
        strat = VIXBasisTrade()
        # Low VIX in contango
        vix_vals = [15 + np.random.normal(0, 1) for _ in range(252)]
        vix_vals[-1] = 14  # current low VIX
        data = {
            "vix_spot": _make_vix_df(vix_vals, "close"),
            "vix3m": _make_vix_df([v * 1.10 for v in vix_vals], "close"),
        }
        signals = strat.generate_signals(data)
        assert len(signals) > 0, "Should generate signal in contango"
        assert signals[0].symbol == "SVXY", "Should trade SVXY in contango"

    def test_crisis_signal(self):
        from strategies.volatility.vix_basis import VIXBasisTrade
        strat = VIXBasisTrade()
        vix_vals = [20] * 251 + [45]  # VIX spike
        data = {
            "vix_spot": _make_vix_df(vix_vals, "close"),
            "vix3m": _make_vix_df([v * 0.95 for v in vix_vals], "close"),
        }
        signals = strat.generate_signals(data)
        assert len(signals) > 0, "Should generate crisis signal"
        assert signals[0].symbol == "UVXY", "Should go long UVXY in crisis"

    def test_regime_classification(self):
        from strategies.volatility.vix_basis import VIXBasisTrade
        strat = VIXBasisTrade()
        regime = strat.classify_regime(12, 14)
        assert regime.level == "low"
        assert regime.term_structure == "contango"

        regime = strat.classify_regime(35, 32)
        assert regime.level == "crisis"
        assert regime.term_structure == "backwardation"


class TestETNCarry:
    """Test ETN Carry strategy."""

    def test_low_vix_signal(self):
        from strategies.volatility.etn_carry import ETNCarryStrategy
        strat = ETNCarryStrategy()
        vix_vals = [14 + np.random.normal(0, 0.5) for _ in range(252)]
        vix_vals[-1] = 13
        data = {"vix": _make_vix_df(vix_vals, "close")}
        signals = strat.generate_signals(data)
        assert len(signals) > 0, "Should generate carry signal in low VIX"

    def test_crisis_protection(self):
        from strategies.volatility.etn_carry import ETNCarryStrategy
        strat = ETNCarryStrategy()
        vix_vals = [20] * 251 + [45]
        data = {"vix": _make_vix_df(vix_vals, "close")}
        signals = strat.generate_signals(data)
        assert any(s.symbol == "UVXY" for s in signals), "Should hedge in crisis"

    def test_risk_checks_cap_weight(self):
        from strategies.volatility.etn_carry import ETNCarryStrategy
        from strategies.base import Signal, SignalDirection
        strat = ETNCarryStrategy()
        signals = [Signal("SVXY", SignalDirection.LONG, 0.10, {"vix": 20})]
        filtered = strat.risk_checks(signals)
        assert len(filtered) == 1, "Should allow SVXY long in normal VIX"
        assert filtered[0].weight <= 0.05, "Should cap weight to max_weight"


class TestVarianceRiskPremium:
    """Test VRP strategy."""

    def test_high_vrp_sell_signal(self):
        from strategies.volatility.variance_risk_premium import VarianceRiskPremium
        strat = VarianceRiskPremium()
        # VIX at 22, RV should be ~14 → VRP ~8
        vix_vals = [20 + np.random.normal(0, 1) for _ in range(252)]
        vix_vals[-1] = 22
        spy_df = _make_spy_df(400, 252)  # low realized vol
        data = {
            "vix": _make_vix_df(vix_vals, "close"),
            "spy": spy_df,
        }
        signals = strat.generate_signals(data)
        # Should generate sell vol signal (VRP > 5)
        assert len(signals) > 0, "Should generate signal when VRP is high"

    def test_realized_vol_computation(self):
        from strategies.volatility.variance_risk_premium import VarianceRiskPremium
        strat = VarianceRiskPremium()
        spy_df = _make_spy_df(400, 252)
        rv = strat.compute_realized_vol(spy_df["close"], 21)
        assert len(rv.dropna()) > 200, "Should compute RV for most of the series"
        assert 5 < float(rv.dropna().mean()) < 40, "RV should be reasonable"

    def test_crisis_blocks_short_vol(self):
        from strategies.volatility.variance_risk_premium import VarianceRiskPremium
        from strategies.base import Signal, SignalDirection
        strat = VarianceRiskPremium()
        signals = [Signal("SPY_IC", SignalDirection.SHORT, 0.03, {"vix": 40})]
        filtered = strat.risk_checks(signals)
        assert len(filtered) == 0, "Should block short vol in crisis"


class TestSkewTrading:
    """Test Skew Trading strategy."""

    def test_steep_skew_signal(self):
        from strategies.volatility.skew_trading import SkewTradingStrategy
        strat = SkewTradingStrategy()
        vix_vals = [18 + np.random.normal(0, 1) for _ in range(252)]
        vix_vals[-1] = 18
        spy_df = _make_spy_df(400, 252)
        # High SKEW index (steep skew)
        skew_vals = [120 + np.random.normal(0, 3) for _ in range(252)]
        skew_vals[-1] = 145  # very steep
        skew_df = pd.DataFrame(
            {"close": skew_vals},
            index=pd.date_range(end=datetime.now(), periods=252, freq="B")
        )
        data = {
            "vix": _make_vix_df(vix_vals, "close"),
            "spy": spy_df,
            "skew": skew_df,
        }
        signals = strat.generate_signals(data)
        assert len(signals) > 0, "Should generate signal on steep skew"

    def test_defensive_mode(self):
        from strategies.volatility.skew_trading import SkewTradingStrategy
        from strategies.base import Signal, SignalDirection
        strat = SkewTradingStrategy()
        signals = [Signal("SPY_PUT_SPREAD", SignalDirection.SHORT, 0.02, {"vix": 35})]
        filtered = strat.risk_checks(signals)
        assert len(filtered) == 0, "Should block short vol in defensive mode"

    def test_skew_estimation(self):
        from strategies.volatility.skew_trading import SkewTradingStrategy
        strat = SkewTradingStrategy()
        idx = pd.date_range(end=datetime.now(), periods=252, freq="B")
        vix = pd.Series([18 + np.random.normal(0, 1) for _ in range(252)], index=idx)
        rets = pd.Series(np.random.normal(0.0003, 0.01, 252), index=idx)
        skew = strat.estimate_skew(vix, rets, window=21)
        assert len(skew.dropna()) > 0, "Should estimate skew for at least some data"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
