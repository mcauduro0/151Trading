"""
151 Trading System — VIX Basis Trade Strategy (VOL_VIX_008)
============================================================
Trades the spread between VIX spot and VIX futures (basis).
When the basis is in steep contango, sells vol (short VIX futures
or long inverse VIX ETNs). When in backwardation, buys vol.

Key signals:
  - VIX futures term structure slope
  - VIX/VIX3M ratio (short-term vs medium-term vol)
  - Contango roll yield capture
  - Regime detection (risk-on vs risk-off)

Data sources: FRED (VIXCLS), Yahoo Finance (VIX futures, UVXY, SVXY)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from strategies.base import StrategyBase, Signal, SignalDirection, AssetClass, StrategyStyle


@dataclass
class VIXRegime:
    """VIX market regime classification."""
    level: str          # "low", "normal", "elevated", "crisis"
    term_structure: str  # "contango", "flat", "backwardation"
    basis_pct: float    # (futures - spot) / spot
    roll_yield: float   # annualized roll yield
    vix_spot: float
    vix_1m: float       # 1-month futures proxy
    vvix: Optional[float] = None  # vol of vol


class VIXBasisTrade(StrategyBase):
    """
    VIX Basis Trade: Captures the roll yield from VIX futures
    term structure contango/backwardation.

    Entry rules:
      - CONTANGO > 5%: Short VIX (sell futures / buy inverse ETN)
      - BACKWARDATION > 3%: Long VIX (buy futures / buy long VIX ETN)
      - FLAT (-3% to +5%): No position

    Risk management:
      - VIX spike > 35: flatten all short vol positions
      - Max position size: 5% of portfolio
      - Stop loss: 15% on any position
      - Rebalance weekly
    """

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(
            strategy_id="VOL_VIX_008",
            name="VIX Basis Trade",
            asset_class=AssetClass.VOLATILITY,
            style=StrategyStyle.CARRY,
            description="Captures roll yield from VIX futures term structure",
        )
        cfg = config or {}
        self.contango_threshold = cfg.get("contango_threshold", 0.05)
        self.backwardation_threshold = cfg.get("backwardation_threshold", -0.03)
        self.vix_crisis_level = cfg.get("vix_crisis_level", 35)
        self.max_position_pct = cfg.get("max_position_pct", 0.05)
        self.lookback_days = cfg.get("lookback_days", 252)
        self.rebalance_freq = cfg.get("rebalance_freq", "weekly")

    def required_data(self) -> Dict[str, str]:
        return {
            "vix_spot": "FRED:VIXCLS",
            "vix_etns": "YAHOO:^VIX,UVXY,SVXY,VXX",
            "vix3m": "FRED:VXVCLS",  # VIX 3-month
            "sp500": "YAHOO:SPY",
        }

    def classify_regime(self, vix_spot: float, vix_1m: float,
                        vix_3m: Optional[float] = None) -> VIXRegime:
        """Classify VIX market regime."""
        # Level classification
        if vix_spot < 15:
            level = "low"
        elif vix_spot < 20:
            level = "normal"
        elif vix_spot < 30:
            level = "elevated"
        else:
            level = "crisis"

        # Basis calculation (1m futures vs spot)
        basis_pct = (vix_1m - vix_spot) / vix_spot if vix_spot > 0 else 0

        # Term structure
        if basis_pct > 0.03:
            term_structure = "contango"
        elif basis_pct < -0.02:
            term_structure = "backwardation"
        else:
            term_structure = "flat"

        # Annualized roll yield (monthly basis * 12)
        roll_yield = basis_pct * 12

        return VIXRegime(
            level=level,
            term_structure=term_structure,
            basis_pct=basis_pct,
            roll_yield=roll_yield,
            vix_spot=vix_spot,
            vix_1m=vix_1m,
            vvix=None,
        )

    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> List[Signal]:
        """Generate VIX basis trade signals."""
        signals = []
        now = datetime.now()

        # Extract VIX data
        vix_df = data.get("vix_spot")
        if vix_df is None or vix_df.empty:
            return signals

        # Get current VIX level
        if isinstance(vix_df, pd.DataFrame):
            vix_spot = float(vix_df.iloc[-1].get("close", vix_df.iloc[-1].get("value", 20)))
        else:
            vix_spot = 20.0

        # Estimate 1-month VIX futures from VIX3M or ETN prices
        vix3m_df = data.get("vix3m")
        if vix3m_df is not None and not vix3m_df.empty:
            vix_3m = float(vix3m_df.iloc[-1].get("close", vix3m_df.iloc[-1].get("value", vix_spot * 1.05)))
        else:
            vix_3m = vix_spot * 1.05  # typical contango assumption

        # Estimate 1-month futures as midpoint between spot and 3M
        vix_1m = (vix_spot + vix_3m) / 2

        # Classify regime
        regime = self.classify_regime(vix_spot, vix_1m, vix_3m)

        # Historical VIX for percentile ranking
        if isinstance(vix_df, pd.DataFrame) and len(vix_df) > 20:
            vix_values = vix_df["close"].dropna().values if "close" in vix_df.columns else vix_df["value"].dropna().values
            vix_percentile = np.sum(vix_values < vix_spot) / len(vix_values)
        else:
            vix_percentile = 0.5

        # Signal generation
        if regime.level == "crisis":
            # Crisis: flatten shorts, potentially go long vol
            signals.append(Signal(
                symbol="UVXY",
                direction=SignalDirection.LONG,
                weight=self.max_position_pct * 0.5,
                metadata={
                    "strategy": self.strategy_id,
                    "regime": regime.level,
                    "vix_spot": vix_spot,
                    "basis_pct": regime.basis_pct,
                    "rationale": f"VIX crisis ({vix_spot:.1f}), long vol protection",
                },
            ))
        elif regime.term_structure == "contango" and regime.basis_pct > self.contango_threshold:
            # Steep contango: short VIX (capture roll yield)
            weight = min(self.max_position_pct, regime.basis_pct * 0.5)
            signals.append(Signal(
                symbol="SVXY",
                direction=SignalDirection.LONG,  # inverse VIX = short vol
                weight=weight,
                metadata={
                    "strategy": self.strategy_id,
                    "regime": regime.level,
                    "term_structure": regime.term_structure,
                    "basis_pct": regime.basis_pct,
                    "roll_yield_ann": regime.roll_yield,
                    "vix_percentile": vix_percentile,
                    "rationale": f"Contango {regime.basis_pct*100:.1f}%, roll yield {regime.roll_yield*100:.1f}% ann",
                },
            ))
        elif regime.term_structure == "backwardation" and regime.basis_pct < self.backwardation_threshold:
            # Backwardation: long VIX
            weight = min(self.max_position_pct * 0.5, abs(regime.basis_pct) * 0.3)
            signals.append(Signal(
                symbol="UVXY",
                direction=SignalDirection.LONG,
                weight=weight,
                metadata={
                    "strategy": self.strategy_id,
                    "regime": regime.level,
                    "term_structure": regime.term_structure,
                    "basis_pct": regime.basis_pct,
                    "rationale": f"Backwardation {regime.basis_pct*100:.1f}%, long vol",
                },
            ))

        return signals

    def risk_checks(self, signals: List[Signal], portfolio_state: Optional[Dict] = None) -> List[Signal]:
        """Apply risk management rules."""
        filtered = []
        for sig in signals:
            meta = sig.metadata or {}
            vix = meta.get("vix_spot", 20)

            # Crisis override: only allow long vol
            if vix > self.vix_crisis_level and sig.direction == SignalDirection.SHORT:
                continue

            # Cap position size
            if sig.weight > self.max_position_pct:
                sig = Signal(
                    symbol=sig.symbol,
                    direction=sig.direction,
                    weight=self.max_position_pct,
                    metadata=sig.metadata,
                )

            filtered.append(sig)
        return filtered
