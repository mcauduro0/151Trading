"""
151 Trading System — Variance Risk Premium Strategy (VOL_VRP_010)
==================================================================
Exploits the persistent premium of implied volatility over realized
volatility. Systematically sells volatility when the VRP is elevated
and buys when compressed.

The Variance Risk Premium (VRP) = Implied Vol - Realized Vol
Historically averages 3-5% annualized, providing a structural edge.

Key signals:
  - VRP level (IV - RV spread)
  - VRP percentile ranking
  - VRP momentum (expanding or compressing)
  - Regime-adjusted thresholds

Data sources: FRED (VIX), Yahoo Finance (SPY for realized vol)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional
from strategies.base import StrategyBase, Signal, SignalDirection, AssetClass, StrategyStyle


class VarianceRiskPremium(StrategyBase):
    """
    Variance Risk Premium: Sells volatility when implied vol is
    significantly above realized vol, capturing the structural
    risk premium that volatility sellers earn.

    VRP = VIX (implied) - Realized Vol (SPY)

    Entry rules:
      - VRP > 5%: Sell vol (short straddle, iron condor, or short VIX)
      - VRP > 8%: Aggressive sell vol
      - VRP < 1%: Buy vol (implied is cheap relative to realized)
      - VRP < -2%: Aggressive buy vol (rare, usually pre-event)

    Implementation:
      - Primary: Iron condors on SPY (defined risk)
      - Secondary: Short VIX futures / long SVXY
      - Hedge: Long OTM puts when VRP is negative

    Risk management:
      - Max loss per trade: 2% of portfolio
      - VIX > 35: no new short vol positions
      - Dynamic delta hedging when |delta| > 0.15
    """

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(
            strategy_id="VOL_VRP_010",
            name="Variance Risk Premium",
            asset_class=AssetClass.VOLATILITY,
            style=StrategyStyle.CARRY,
            description="Captures the spread between implied and realized volatility",
        )
        cfg = config or {}
        self.rv_window = cfg.get("rv_window", 21)  # 21-day realized vol
        self.vrp_sell_threshold = cfg.get("vrp_sell_threshold", 5.0)
        self.vrp_buy_threshold = cfg.get("vrp_buy_threshold", 1.0)
        self.vrp_aggressive_sell = cfg.get("vrp_aggressive_sell", 8.0)
        self.vrp_aggressive_buy = cfg.get("vrp_aggressive_buy", -2.0)
        self.max_weight = cfg.get("max_weight", 0.04)
        self.vix_hard_stop = cfg.get("vix_hard_stop", 35)
        self.lookback = cfg.get("lookback", 252)

    def required_data(self) -> Dict[str, str]:
        return {
            "vix": "FRED:VIXCLS",
            "spy": "YAHOO:SPY",
        }

    def compute_realized_vol(self, prices: pd.Series, window: int = 21) -> pd.Series:
        """
        Compute annualized realized volatility using close-to-close returns.

        Args:
            prices: Price series
            window: Rolling window in trading days

        Returns:
            Annualized realized vol series (in percentage points)
        """
        log_returns = np.log(prices / prices.shift(1))
        rv = log_returns.rolling(window).std() * np.sqrt(252) * 100
        return rv

    def compute_vrp(self, vix_series: pd.Series, rv_series: pd.Series) -> pd.Series:
        """
        Compute Variance Risk Premium = Implied Vol (VIX) - Realized Vol.
        """
        # Align indices
        aligned = pd.concat([vix_series, rv_series], axis=1, join="inner")
        aligned.columns = ["iv", "rv"]
        vrp = aligned["iv"] - aligned["rv"]
        return vrp

    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> List[Signal]:
        """Generate VRP-based trading signals."""
        signals = []

        # Get VIX (implied vol)
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

        # Get SPY prices for realized vol
        spy_df = data.get("spy")
        if spy_df is None or spy_df.empty:
            return signals

        spy_close = spy_df["close"].dropna() if "close" in spy_df.columns else None
        if spy_close is None or len(spy_close) < self.rv_window + 5:
            return signals

        # Compute realized vol
        rv_series = self.compute_realized_vol(spy_close, self.rv_window)
        rv_current = float(rv_series.iloc[-1]) if not rv_series.empty else vix_current

        # Compute VRP
        vrp_current = vix_current - rv_current

        # Historical VRP for percentile
        vrp_series = self.compute_vrp(vix_series, rv_series)
        if len(vrp_series) > 20:
            vrp_percentile = float(np.sum(vrp_series.values < vrp_current) / len(vrp_series))
            vrp_mean = float(vrp_series.tail(self.lookback).mean())
            vrp_std = float(vrp_series.tail(self.lookback).std())
            vrp_zscore = (vrp_current - vrp_mean) / vrp_std if vrp_std > 0 else 0
        else:
            vrp_percentile = 0.5
            vrp_mean = 4.0
            vrp_zscore = 0

        # VRP momentum (5-day change)
        if len(vrp_series) >= 5:
            vrp_5d_ago = float(vrp_series.iloc[-5])
            vrp_momentum = vrp_current - vrp_5d_ago
        else:
            vrp_momentum = 0

        # Signal generation
        base_meta = {
            "strategy": self.strategy_id,
            "vix": round(vix_current, 2),
            "rv_21d": round(rv_current, 2),
            "vrp": round(vrp_current, 2),
            "vrp_percentile": round(vrp_percentile, 2),
            "vrp_zscore": round(vrp_zscore, 2),
            "vrp_momentum_5d": round(vrp_momentum, 2),
        }

        if vix_current > self.vix_hard_stop:
            # Crisis: buy vol for protection
            signals.append(Signal(
                symbol="SPY_PUT",
                direction=SignalDirection.LONG,
                weight=self.max_weight * 0.5,
                metadata={
                    **base_meta,
                    "regime": "crisis",
                    "rationale": f"VIX {vix_current:.1f} > {self.vix_hard_stop}, long vol protection",
                },
            ))
        elif vrp_current >= self.vrp_aggressive_sell:
            # Aggressive sell vol: VRP very wide
            weight = min(self.max_weight, 0.03 + (vrp_current - self.vrp_sell_threshold) * 0.002)
            signals.append(Signal(
                symbol="SPY_IC",  # Iron Condor on SPY
                direction=SignalDirection.SHORT,
                weight=weight,
                metadata={
                    **base_meta,
                    "structure": "iron_condor",
                    "regime": "high_vrp",
                    "rationale": f"VRP {vrp_current:.1f}% (p{vrp_percentile*100:.0f}), aggressive sell vol",
                },
            ))
        elif vrp_current >= self.vrp_sell_threshold:
            # Normal sell vol
            weight = self.max_weight * 0.6
            signals.append(Signal(
                symbol="SPY_IC",
                direction=SignalDirection.SHORT,
                weight=weight,
                metadata={
                    **base_meta,
                    "structure": "iron_condor",
                    "regime": "normal_vrp",
                    "rationale": f"VRP {vrp_current:.1f}% > {self.vrp_sell_threshold}%, sell vol",
                },
            ))
        elif vrp_current <= self.vrp_aggressive_buy:
            # Rare: implied is cheap vs realized → buy vol
            signals.append(Signal(
                symbol="SPY_STRADDLE",
                direction=SignalDirection.LONG,
                weight=self.max_weight * 0.4,
                metadata={
                    **base_meta,
                    "structure": "long_straddle",
                    "regime": "negative_vrp",
                    "rationale": f"VRP {vrp_current:.1f}% < {self.vrp_aggressive_buy}%, buy vol",
                },
            ))
        elif vrp_current <= self.vrp_buy_threshold:
            # Compressed VRP: cautious buy vol
            signals.append(Signal(
                symbol="SPY_STRANGLE",
                direction=SignalDirection.LONG,
                weight=self.max_weight * 0.25,
                metadata={
                    **base_meta,
                    "structure": "long_strangle",
                    "regime": "compressed_vrp",
                    "rationale": f"VRP {vrp_current:.1f}% compressed, buy vol",
                },
            ))

        return signals

    def risk_checks(self, signals: List[Signal], portfolio_state: Optional[Dict] = None) -> List[Signal]:
        """Apply VRP strategy risk management."""
        filtered = []
        for sig in signals:
            meta = sig.metadata or {}
            vix = meta.get("vix", 20)

            # No short vol in crisis
            if vix > self.vix_hard_stop and sig.direction == SignalDirection.SHORT:
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
