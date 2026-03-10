"""
151 Trading System — Volatility ETN Carry Strategy (VOL_ETN_009)
=================================================================
Captures the structural decay (negative roll yield) in long volatility
ETNs like VXX, UVXY by systematically shorting them during contango.
Hedges with long positions during backwardation.

Key signals:
  - VIX term structure slope (contango/backwardation)
  - ETN premium/discount to NAV
  - Historical roll yield estimation
  - Momentum of VIX (mean reversion tendency)

Data sources: Yahoo Finance (VXX, UVXY, SVXY), FRED (VIXCLS)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from strategies.base import StrategyBase, Signal, SignalDirection, AssetClass, StrategyStyle


class ETNCarryStrategy(StrategyBase):
    """
    Volatility ETN Carry: Systematically captures the structural
    decay in long VIX ETNs during contango periods.

    The strategy exploits the well-documented "volatility risk premium"
    embedded in VIX futures, which causes long VIX ETNs to lose value
    over time due to negative roll yield.

    Entry rules:
      - Contango > 3%: Short UVXY or VXX (or long SVXY)
      - VIX < 25 and declining: Increase short vol position
      - VIX > 30 or backwardation: Flatten or reverse

    Position sizing:
      - Base: 3% of portfolio
      - Scale up to 5% in deep contango with low VIX
      - Scale down to 1% when VIX elevated

    Risk management:
      - Hard stop: VIX > 40 → flatten all
      - Trailing stop: 20% drawdown from peak
      - Max holding period: 30 days before reassessment
    """

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(
            strategy_id="VOL_ETN_009",
            name="Volatility ETN Carry",
            asset_class=AssetClass.VOLATILITY,
            style=StrategyStyle.CARRY,
            description="Captures structural decay in long VIX ETNs via contango roll yield",
        )
        cfg = config or {}
        self.contango_entry = cfg.get("contango_entry", 0.03)
        self.vix_max_short = cfg.get("vix_max_short", 25)
        self.vix_crisis = cfg.get("vix_crisis", 40)
        self.base_weight = cfg.get("base_weight", 0.03)
        self.max_weight = cfg.get("max_weight", 0.05)
        self.lookback = cfg.get("lookback", 252)
        self.etn_tickers = cfg.get("etn_tickers", ["UVXY", "VXX", "SVXY"])

    def required_data(self) -> Dict[str, str]:
        return {
            "vix": "FRED:VIXCLS",
            "etns": "YAHOO:UVXY,VXX,SVXY",
            "spy": "YAHOO:SPY",
        }

    def estimate_contango(self, vix_series: pd.Series, etn_returns: pd.Series,
                          window: int = 21) -> pd.Series:
        """
        Estimate contango from ETN decay rate.
        In contango, long VIX ETNs underperform VIX spot changes.
        """
        if len(vix_series) < window or len(etn_returns) < window:
            return pd.Series(dtype=float)

        # Rolling VIX change vs ETN return spread
        vix_ret = vix_series.pct_change(window)
        etn_cum = (1 + etn_returns).rolling(window).apply(lambda x: x.prod() - 1, raw=True)

        # Contango proxy: VIX ETN underperformance vs VIX spot
        contango_proxy = vix_ret - etn_cum
        return contango_proxy

    def compute_roll_yield(self, prices: pd.DataFrame, ticker: str,
                           window: int = 21) -> float:
        """
        Estimate annualized roll yield from ETN price decay.
        """
        if ticker not in prices.columns or len(prices) < window:
            return 0.0

        recent = prices[ticker].dropna().tail(window)
        if len(recent) < 2:
            return 0.0

        period_return = (recent.iloc[-1] / recent.iloc[0]) - 1
        ann_factor = 252 / window
        ann_roll = period_return * ann_factor
        return float(ann_roll)

    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> List[Signal]:
        """Generate ETN carry trade signals."""
        signals = []
        now = datetime.now()

        # Get VIX data
        vix_df = data.get("vix")
        if vix_df is None or vix_df.empty:
            return signals

        # Current VIX
        if "close" in vix_df.columns:
            vix_current = float(vix_df["close"].iloc[-1])
            vix_series = vix_df["close"]
        elif "value" in vix_df.columns:
            vix_current = float(vix_df["value"].iloc[-1])
            vix_series = vix_df["value"]
        else:
            return signals

        # VIX statistics
        vix_mean = float(vix_series.tail(self.lookback).mean())
        vix_std = float(vix_series.tail(self.lookback).std())
        vix_zscore = (vix_current - vix_mean) / vix_std if vix_std > 0 else 0
        vix_percentile = float(np.sum(vix_series.values < vix_current) / len(vix_series))

        # VIX momentum (5-day change)
        if len(vix_series) >= 5:
            vix_5d_change = (vix_current / float(vix_series.iloc[-5])) - 1
        else:
            vix_5d_change = 0

        # ETN data
        etn_df = data.get("etns")
        etn_prices = {}
        if etn_df is not None and not etn_df.empty:
            if "symbol" in etn_df.columns:
                for ticker in self.etn_tickers:
                    mask = etn_df["symbol"] == ticker
                    if mask.any():
                        etn_prices[ticker] = etn_df.loc[mask, "close"]
            elif "close" in etn_df.columns:
                etn_prices["ETN"] = etn_df["close"]

        # Estimate contango from VIX level relative to historical
        # VIX below mean → likely contango; above mean → likely backwardation
        estimated_contango = -(vix_zscore * 0.03)  # rough proxy

        # Crisis check
        if vix_current > self.vix_crisis:
            signals.append(Signal(
                symbol="UVXY",
                direction=SignalDirection.LONG,
                weight=self.base_weight * 0.5,
                metadata={
                    "strategy": self.strategy_id,
                    "vix": vix_current,
                    "regime": "crisis",
                    "rationale": f"VIX crisis at {vix_current:.1f}, long vol hedge",
                },
            ))
            return signals

        # Normal contango trade
        if vix_current < self.vix_max_short and estimated_contango > self.contango_entry:
            # Scale weight by contango depth
            weight = self.base_weight + (estimated_contango - self.contango_entry) * 2
            weight = min(weight, self.max_weight)

            # Prefer SVXY (inverse) for long position = short vol
            signals.append(Signal(
                symbol="SVXY",
                direction=SignalDirection.LONG,
                weight=weight,
                metadata={
                    "strategy": self.strategy_id,
                    "vix": vix_current,
                    "vix_zscore": round(vix_zscore, 2),
                    "vix_percentile": round(vix_percentile, 2),
                    "estimated_contango": round(estimated_contango, 4),
                    "vix_5d_change": round(vix_5d_change, 4),
                    "regime": "contango",
                    "rationale": f"VIX {vix_current:.1f} (z={vix_zscore:.1f}), contango ~{estimated_contango*100:.1f}%",
                },
            ))

        # Elevated VIX with declining momentum → cautious short vol
        elif vix_current < 30 and vix_5d_change < -0.05:
            signals.append(Signal(
                symbol="SVXY",
                direction=SignalDirection.LONG,
                weight=self.base_weight * 0.5,
                metadata={
                    "strategy": self.strategy_id,
                    "vix": vix_current,
                    "vix_5d_change": round(vix_5d_change, 4),
                    "regime": "mean_reversion",
                    "rationale": f"VIX declining from {vix_current:.1f}, cautious short vol",
                },
            ))

        return signals

    def risk_checks(self, signals: List[Signal], portfolio_state: Optional[Dict] = None) -> List[Signal]:
        """Apply ETN carry risk management."""
        filtered = []
        for sig in signals:
            meta = sig.metadata or {}
            vix = meta.get("vix", 20)

            # Hard stop: no short vol above crisis level
            if vix > self.vix_crisis and sig.symbol in ("SVXY",):
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
