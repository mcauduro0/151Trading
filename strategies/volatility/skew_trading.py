"""
151 Trading System — Volatility Skew Trading Strategy (VOL_SKEW_011)
=====================================================================
Trades the volatility skew — the difference in implied volatility
between OTM puts and OTM calls on the same underlying.

The skew reflects market fear: steep skew = expensive downside protection.
Strategy profits when skew mean-reverts or when skew signals are
predictive of future returns.

Key signals:
  - Put-call skew level (25-delta put IV vs 25-delta call IV)
  - Skew percentile ranking (historical context)
  - Skew momentum (steepening or flattening)
  - Skew vs VIX divergence

Data sources: FRED (VIXCLS), Yahoo Finance (SPY), derived skew metrics
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional
from strategies.base import StrategyBase, Signal, SignalDirection, AssetClass, StrategyStyle


class SkewTradingStrategy(StrategyBase):
    """
    Volatility Skew Trading: Exploits mean-reversion and predictive
    signals in the options volatility skew.

    Skew = IV(25-delta put) - IV(25-delta call)

    When skew is steep (expensive puts):
      - Sell put spreads (collect rich premium)
      - Buy risk reversals (short put, long call)

    When skew is flat (cheap puts):
      - Buy put protection (insurance is cheap)
      - Sell call spreads

    Implementation:
      - Primary: Risk reversals on SPY
      - Secondary: Put spread selling/buying
      - Tertiary: Butterfly adjustments based on skew

    Risk management:
      - Max position: 3% of portfolio
      - Skew > 2 std: reduce position (extreme fear)
      - VIX > 30: only defensive trades
    """

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(
            strategy_id="VOL_SKEW_011",
            name="Volatility Skew Trading",
            asset_class=AssetClass.VOLATILITY,
            style=StrategyStyle.MEAN_REVERSION,
            description="Trades mean-reversion in options volatility skew",
        )
        cfg = config or {}
        self.skew_lookback = cfg.get("skew_lookback", 252)
        self.steep_threshold_pct = cfg.get("steep_threshold_pct", 75)
        self.flat_threshold_pct = cfg.get("flat_threshold_pct", 25)
        self.max_weight = cfg.get("max_weight", 0.03)
        self.vix_defensive = cfg.get("vix_defensive", 30)

    def required_data(self) -> Dict[str, str]:
        return {
            "vix": "FRED:VIXCLS",
            "spy": "YAHOO:SPY",
            "skew": "YAHOO:^SKEW",  # CBOE SKEW Index
        }

    def estimate_skew(self, vix_series: pd.Series, spy_returns: pd.Series,
                      window: int = 21) -> pd.Series:
        """
        Estimate volatility skew from VIX and realized return distribution.

        Uses the relationship between VIX level, realized skewness of returns,
        and the CBOE SKEW index when available.

        Higher VIX + negative return skewness → steeper skew
        """
        # Realized return skewness (rolling)
        ret_skew = spy_returns.rolling(window).skew()

        # VIX level contribution
        vix_normalized = (vix_series - vix_series.rolling(self.skew_lookback).mean()) / \
                         vix_series.rolling(self.skew_lookback).std()

        # Composite skew estimate
        # Higher VIX z-score + more negative return skew → steeper implied skew
        skew_estimate = vix_normalized * 2 - ret_skew * 3 + 10  # base skew ~10 vol points
        return skew_estimate

    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> List[Signal]:
        """Generate skew trading signals."""
        signals = []

        # VIX data
        vix_df = data.get("vix")
        if vix_df is None or vix_df.empty:
            return signals

        if "close" in vix_df.columns:
            vix_series = vix_df["close"].dropna()
        elif "value" in vix_df.columns:
            vix_series = vix_df["value"].dropna()
        else:
            return signals

        vix_current = float(vix_series.iloc[-1])

        # SPY data for realized metrics
        spy_df = data.get("spy")
        if spy_df is None or spy_df.empty or len(spy_df) < 30:
            return signals

        spy_close = spy_df["close"].dropna()
        spy_returns = spy_close.pct_change().dropna()

        # CBOE SKEW Index (if available)
        skew_df = data.get("skew")
        if skew_df is not None and not skew_df.empty and "close" in skew_df.columns:
            skew_series = skew_df["close"].dropna()
            skew_current = float(skew_series.iloc[-1])
            use_cboe_skew = True
        else:
            # Estimate skew from VIX and returns
            skew_series = self.estimate_skew(vix_series, spy_returns)
            skew_current = float(skew_series.iloc[-1]) if not skew_series.empty else 10
            use_cboe_skew = False

        # Skew statistics
        if len(skew_series) > 20:
            skew_mean = float(skew_series.tail(self.skew_lookback).mean())
            skew_std = float(skew_series.tail(self.skew_lookback).std())
            skew_zscore = (skew_current - skew_mean) / skew_std if skew_std > 0 else 0
            skew_percentile = float(np.sum(skew_series.values < skew_current) / len(skew_series)) * 100
        else:
            skew_mean = skew_current
            skew_zscore = 0
            skew_percentile = 50

        # Skew momentum (5-day)
        if len(skew_series) >= 5:
            skew_5d_change = skew_current - float(skew_series.iloc[-5])
        else:
            skew_5d_change = 0

        # Realized return skewness
        ret_skewness = float(spy_returns.tail(21).skew()) if len(spy_returns) >= 21 else 0

        base_meta = {
            "strategy": self.strategy_id,
            "vix": round(vix_current, 2),
            "skew": round(skew_current, 2),
            "skew_source": "CBOE" if use_cboe_skew else "estimated",
            "skew_percentile": round(skew_percentile, 1),
            "skew_zscore": round(skew_zscore, 2),
            "skew_5d_change": round(skew_5d_change, 2),
            "ret_skewness_21d": round(ret_skewness, 3),
        }

        # Signal generation based on skew regime
        if vix_current > self.vix_defensive:
            # Defensive mode: only buy protection if skew is flat (cheap puts)
            if skew_percentile < self.flat_threshold_pct:
                signals.append(Signal(
                    symbol="SPY_PUT_SPREAD",
                    direction=SignalDirection.LONG,
                    weight=self.max_weight * 0.5,
                    metadata={
                        **base_meta,
                        "structure": "long_put_spread",
                        "regime": "defensive_flat_skew",
                        "rationale": f"VIX elevated ({vix_current:.1f}), skew flat (p{skew_percentile:.0f}), cheap protection",
                    },
                ))
        elif skew_percentile > self.steep_threshold_pct:
            # Steep skew: sell puts (expensive), buy calls (cheap)
            weight = self.max_weight * min(1.0, (skew_percentile - self.steep_threshold_pct) / 25)

            # Risk reversal: short put + long call
            signals.append(Signal(
                symbol="SPY_RISK_REVERSAL",
                direction=SignalDirection.LONG,
                weight=weight,
                metadata={
                    **base_meta,
                    "structure": "risk_reversal",
                    "regime": "steep_skew",
                    "rationale": f"Skew steep (p{skew_percentile:.0f}, z={skew_zscore:.1f}), sell expensive puts",
                },
            ))

            # Also sell put spreads
            if skew_percentile > 85:
                signals.append(Signal(
                    symbol="SPY_PUT_SPREAD",
                    direction=SignalDirection.SHORT,
                    weight=weight * 0.5,
                    metadata={
                        **base_meta,
                        "structure": "short_put_spread",
                        "regime": "very_steep_skew",
                        "rationale": f"Skew very steep (p{skew_percentile:.0f}), sell put spreads",
                    },
                ))

        elif skew_percentile < self.flat_threshold_pct:
            # Flat skew: buy puts (cheap protection), sell calls
            weight = self.max_weight * min(1.0, (self.flat_threshold_pct - skew_percentile) / 25)

            signals.append(Signal(
                symbol="SPY_PUT_SPREAD",
                direction=SignalDirection.LONG,
                weight=weight,
                metadata={
                    **base_meta,
                    "structure": "long_put_spread",
                    "regime": "flat_skew",
                    "rationale": f"Skew flat (p{skew_percentile:.0f}), cheap put protection",
                },
            ))

        # Skew mean-reversion signal (extreme z-scores)
        if abs(skew_zscore) > 2.0:
            direction = SignalDirection.SHORT if skew_zscore > 0 else SignalDirection.LONG
            signals.append(Signal(
                symbol="SPY_BUTTERFLY",
                direction=direction,
                weight=self.max_weight * 0.3,
                metadata={
                    **base_meta,
                    "structure": "butterfly",
                    "regime": "skew_extreme",
                    "rationale": f"Skew extreme (z={skew_zscore:.1f}), mean-reversion trade",
                },
            ))

        return signals

    def risk_checks(self, signals: List[Signal], portfolio_state: Optional[Dict] = None) -> List[Signal]:
        """Apply skew strategy risk management."""
        filtered = []
        for sig in signals:
            meta = sig.metadata or {}
            vix = meta.get("vix", 20)

            # No selling vol in crisis
            if vix > self.vix_defensive and sig.direction == SignalDirection.SHORT:
                continue

            # Cap weight
            if sig.weight > self.max_weight:
                sig = Signal(
                    symbol=sig.symbol,
                    direction=sig.direction,
                    weight=self.max_weight,
                    metadata=sig.metadata,
                )

            filtered.append(sig)
        return filtered
