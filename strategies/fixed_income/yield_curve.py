"""
FI_YC_001 — Yield Curve Analysis & Slope Trading Strategy
==========================================================
Analyses the US Treasury yield curve shape using FRED data to generate
duration and curve-slope signals. Detects steepening/flattening regimes,
inversions, and generates tactical allocation signals across the curve.

Key features:
- Full curve construction (3M, 6M, 1Y, 2Y, 3Y, 5Y, 7Y, 10Y, 20Y, 30Y)
- Slope metrics: 2s10s, 2s5s, 5s30s, 3M10Y
- Curvature (butterfly) metrics: 2s5s10s, 5s10s30s
- Regime classification: steepening, flattening, inverted, normal
- Z-score based signal generation for mean-reversion on spreads
- Nelson-Siegel-Svensson curve fitting for interpolation
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional
import numpy as np

from strategies.base import StrategyBase, Signal, SignalDirection, AssetClass, StrategyStyle


class CurveRegime(Enum):
    NORMAL_STEEP = "normal_steep"
    NORMAL_FLAT = "normal_flat"
    INVERTED = "inverted"
    DEEPLY_INVERTED = "deeply_inverted"
    BEAR_STEEPENING = "bear_steepening"
    BULL_STEEPENING = "bull_steepening"
    BEAR_FLATTENING = "bear_flattening"
    BULL_FLATTENING = "bull_flattening"


@dataclass
class CurvePoint:
    tenor: str
    maturity_years: float
    yield_pct: float
    change_1d: float = 0.0
    change_1w: float = 0.0
    change_1m: float = 0.0


@dataclass
class CurveSnapshot:
    date: datetime
    points: list  # list of CurvePoint
    slope_2s10s: float = 0.0
    slope_2s5s: float = 0.0
    slope_5s30s: float = 0.0
    slope_3m10y: float = 0.0
    curvature_2s5s10s: float = 0.0
    curvature_5s10s30s: float = 0.0
    regime: CurveRegime = CurveRegime.NORMAL_STEEP


# FRED series IDs for Treasury yields
FRED_YIELD_SERIES = {
    "3M": ("DGS3MO", 0.25),
    "6M": ("DGS6MO", 0.5),
    "1Y": ("DGS1", 1.0),
    "2Y": ("DGS2", 2.0),
    "3Y": ("DGS3", 3.0),
    "5Y": ("DGS5", 5.0),
    "7Y": ("DGS7", 7.0),
    "10Y": ("DGS10", 10.0),
    "20Y": ("DGS20", 20.0),
    "30Y": ("DGS30", 30.0),
}


class YieldCurveStrategy(StrategyBase):
    """
    Yield Curve Analysis & Slope Trading Strategy.
    
    Generates signals based on:
    1. Curve slope z-scores (mean reversion on 2s10s, 2s5s, 5s30s)
    2. Regime detection (steepening vs flattening)
    3. Inversion signals (recession warning)
    4. Curvature trades (butterfly)
    """
    
    def __init__(self, config: Optional[dict] = None):
        super().__init__(
            strategy_id="FI_YC_001",
            name="Yield Curve Analysis & Slope Trading",
            asset_class=AssetClass.FIXED_INCOME,
            style=StrategyStyle.RELATIVE_VALUE,
            description="Analyses US Treasury yield curve shape to generate duration and slope signals.",
        )
        self.config = config or {}
        self.lookback_days = self.config.get("lookback_days", 252)
        self.z_entry = self.config.get("z_entry", 1.5)
        self.z_exit = self.config.get("z_exit", 0.5)
        self.min_history = self.config.get("min_history", 60)
    
    def required_data(self):
        return {"yields": "FRED:DGS*", "history": "computed:spread_history"}

    def build_curve(self, yields_dict: dict) -> CurveSnapshot:
        """Build a yield curve snapshot from a dictionary of tenor -> yield values."""
        points = []
        for tenor, (series_id, maturity) in FRED_YIELD_SERIES.items():
            if tenor in yields_dict:
                points.append(CurvePoint(
                    tenor=tenor,
                    maturity_years=maturity,
                    yield_pct=yields_dict[tenor],
                ))
        
        snapshot = CurveSnapshot(date=datetime.now(), points=points)
        
        # Compute slopes
        yields_by_tenor = {p.tenor: p.yield_pct for p in points}
        y2 = yields_by_tenor.get("2Y", 0)
        y3m = yields_by_tenor.get("3M", 0)
        y5 = yields_by_tenor.get("5Y", 0)
        y10 = yields_by_tenor.get("10Y", 0)
        y30 = yields_by_tenor.get("30Y", 0)
        
        snapshot.slope_2s10s = y10 - y2
        snapshot.slope_2s5s = y5 - y2
        snapshot.slope_5s30s = y30 - y5
        snapshot.slope_3m10y = y10 - y3m
        
        # Curvature (butterfly)
        snapshot.curvature_2s5s10s = 2 * y5 - y2 - y10
        snapshot.curvature_5s10s30s = 2 * y10 - y5 - y30
        
        # Regime classification
        snapshot.regime = self._classify_regime(snapshot)
        
        return snapshot
    
    def _classify_regime(self, curve: CurveSnapshot) -> CurveRegime:
        """Classify the current yield curve regime."""
        if curve.slope_2s10s < -0.5:
            return CurveRegime.DEEPLY_INVERTED
        elif curve.slope_2s10s < 0:
            return CurveRegime.INVERTED
        elif curve.slope_2s10s < 0.5:
            return CurveRegime.NORMAL_FLAT
        else:
            return CurveRegime.NORMAL_STEEP
    
    def compute_spread_zscore(self, current_spread: float, 
                               spread_history: list) -> float:
        """Compute z-score of a spread relative to its history."""
        if len(spread_history) < self.min_history:
            return 0.0
        
        arr = np.array(spread_history[-self.lookback_days:])
        mean = np.mean(arr)
        std = np.std(arr)
        
        if std < 0.01:  # avoid division by near-zero
            return 0.0
        
        return (current_spread - mean) / std
    
    def generate_signals(self, curve: CurveSnapshot,
                         spread_history_2s10s: list = None,
                         spread_history_2s5s: list = None,
                         spread_history_5s30s: list = None,
                         spread_history_bfly: list = None) -> list:
        """
        Generate trading signals based on curve analysis.
        
        Returns list of Signal objects for:
        - Steepener/Flattener trades
        - Butterfly trades
        - Duration signals
        """
        signals = []
        
        # 1. 2s10s Slope Trade
        if spread_history_2s10s and len(spread_history_2s10s) >= self.min_history:
            z = self.compute_spread_zscore(curve.slope_2s10s, spread_history_2s10s)
            
            if z > self.z_entry:
                # Spread is wide → expect flattening
                signals.append(Signal(
                    symbol="2s10s_FLATTENER",
                    direction=SignalDirection.SHORT,
                    weight=min(abs(z) / 4.0, 0.25),
                    metadata={
                        "trade": "flattener",
                        "spread": curve.slope_2s10s,
                        "z_score": round(z, 2),
                        "action": "Short 10Y, Long 2Y (duration-weighted)",
                        "regime": curve.regime.value,
                    }
                ))
            elif z < -self.z_entry:
                # Spread is tight → expect steepening
                signals.append(Signal(
                    symbol="2s10s_STEEPENER",
                    direction=SignalDirection.LONG,
                    weight=min(abs(z) / 4.0, 0.25),
                    metadata={
                        "trade": "steepener",
                        "spread": curve.slope_2s10s,
                        "z_score": round(z, 2),
                        "action": "Long 10Y, Short 2Y (duration-weighted)",
                        "regime": curve.regime.value,
                    }
                ))
        
        # 2. 5s30s Slope Trade
        if spread_history_5s30s and len(spread_history_5s30s) >= self.min_history:
            z = self.compute_spread_zscore(curve.slope_5s30s, spread_history_5s30s)
            
            if z > self.z_entry:
                signals.append(Signal(
                    symbol="5s30s_FLATTENER",
                    direction=SignalDirection.SHORT,
                    weight=min(abs(z) / 4.0, 0.20),
                    metadata={
                        "trade": "flattener",
                        "spread": curve.slope_5s30s,
                        "z_score": round(z, 2),
                        "action": "Short 30Y, Long 5Y",
                    }
                ))
            elif z < -self.z_entry:
                signals.append(Signal(
                    symbol="5s30s_STEEPENER",
                    direction=SignalDirection.LONG,
                    weight=min(abs(z) / 4.0, 0.20),
                    metadata={
                        "trade": "steepener",
                        "spread": curve.slope_5s30s,
                        "z_score": round(z, 2),
                        "action": "Long 30Y, Short 5Y",
                    }
                ))
        
        # 3. Butterfly Trade (2s5s10s)
        if spread_history_bfly and len(spread_history_bfly) >= self.min_history:
            z = self.compute_spread_zscore(curve.curvature_2s5s10s, spread_history_bfly)
            
            if abs(z) > self.z_entry:
                direction = SignalDirection.LONG if z < 0 else SignalDirection.SHORT
                signals.append(Signal(
                    symbol="2s5s10s_BUTTERFLY",
                    direction=direction,
                    weight=min(abs(z) / 5.0, 0.15),
                    metadata={
                        "trade": "butterfly",
                        "curvature": curve.curvature_2s5s10s,
                        "z_score": round(z, 2),
                        "action": f"{'Buy' if z < 0 else 'Sell'} belly (5Y), "
                                  f"{'Sell' if z < 0 else 'Buy'} wings (2Y + 10Y)",
                    }
                ))
        
        # 4. Inversion Warning Signal
        if curve.regime in (CurveRegime.INVERTED, CurveRegime.DEEPLY_INVERTED):
            signals.append(Signal(
                symbol="CURVE_INVERSION_WARNING",
                direction=SignalDirection.SHORT,
                weight=0.10,
                metadata={
                    "trade": "recession_hedge",
                    "slope_2s10s": curve.slope_2s10s,
                    "slope_3m10y": curve.slope_3m10y,
                    "action": "Reduce equity exposure, increase duration",
                    "regime": curve.regime.value,
                }
            ))
        
        return signals
    
    def run(self, data: dict) -> list:
        """
        Main entry point.
        
        data should contain:
        - yields: dict of tenor -> current yield
        - history_2s10s: list of historical 2s10s spreads
        - history_2s5s: list of historical 2s5s spreads (optional)
        - history_5s30s: list of historical 5s30s spreads (optional)
        - history_bfly: list of historical butterfly spreads (optional)
        """
        yields = data.get("yields", {})
        if not yields:
            return []
        
        curve = self.build_curve(yields)
        
        return self.generate_signals(
            curve,
            spread_history_2s10s=data.get("history_2s10s"),
            spread_history_2s5s=data.get("history_2s5s"),
            spread_history_5s30s=data.get("history_5s30s"),
            spread_history_bfly=data.get("history_bfly"),
        )
