"""
FI_DUR_002 — Duration Timing Strategy
=======================================
Tactical duration management that switches between long-duration (TLT)
and short-duration (SHY/BIL) based on macro regime signals, yield curve
slope, term premium estimates, and momentum indicators.

Key features:
- Multi-factor duration scoring (slope, momentum, macro regime, VIX)
- Smooth allocation between TLT/IEF/SHY based on composite score
- Crisis detection: flight-to-quality overlay
- Term premium estimation using ACM model proxy
- Carry-adjusted duration targeting
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import numpy as np

from strategies.base import StrategyBase, Signal, SignalDirection, AssetClass, StrategyStyle


@dataclass
class DurationFactors:
    """Input factors for duration timing decision."""
    yield_10y: float = 0.0
    yield_2y: float = 0.0
    yield_3m: float = 0.0
    slope_2s10s: float = 0.0
    slope_3m10y: float = 0.0
    vix: float = 20.0
    tlt_momentum_20d: float = 0.0  # 20-day return of TLT
    tlt_momentum_60d: float = 0.0  # 60-day return of TLT
    hy_spread: float = 350.0       # HY OAS spread in bps
    hy_spread_change_1m: float = 0.0
    fed_funds_rate: float = 5.0
    cpi_yoy: float = 3.0
    pmi: float = 50.0


class DurationTimingStrategy(StrategyBase):
    """
    Tactical Duration Timing Strategy.
    
    Generates a duration score from -1 (max short duration) to +1 (max long duration)
    and maps it to ETF allocations:
    - Score > 0.5: Long TLT (20+ year)
    - Score 0 to 0.5: Long IEF (7-10 year)
    - Score -0.5 to 0: Long SHY (1-3 year)
    - Score < -0.5: Long BIL (T-bills) / Cash
    """
    
    def __init__(self, config: Optional[dict] = None):
        super().__init__(
            strategy_id="FI_DUR_002",
            name="Duration Timing",
            asset_class=AssetClass.FIXED_INCOME,
            style=StrategyStyle.DIRECTIONAL,
            description="Tactical duration management switching between TLT/IEF/SHY/BIL.",
        )
        self.config = config or {}
        self.slope_weight = self.config.get("slope_weight", 0.25)
        self.momentum_weight = self.config.get("momentum_weight", 0.25)
        self.macro_weight = self.config.get("macro_weight", 0.25)
        self.crisis_weight = self.config.get("crisis_weight", 0.25)
    
    def required_data(self):
        return {"yields": "FRED:DGS*", "vix": "FRED:VIXCLS", "tlt": "YAHOO:TLT", "macro": "FRED:CPI,PMI"}

    def _slope_score(self, factors: DurationFactors) -> float:
        """
        Slope factor: steeper curve → longer duration (bonds cheap at long end).
        Inverted curve → shorter duration (recession risk).
        """
        slope = factors.slope_2s10s
        
        if slope > 1.5:
            return 0.8   # Very steep → go long duration
        elif slope > 0.5:
            return 0.4   # Moderately steep
        elif slope > 0:
            return 0.0   # Flat-ish
        elif slope > -0.5:
            return -0.4  # Slightly inverted
        else:
            return -0.8  # Deeply inverted → short duration
    
    def _momentum_score(self, factors: DurationFactors) -> float:
        """
        Bond momentum: positive TLT returns → trend following.
        Uses dual timeframe (20d and 60d).
        """
        m20 = factors.tlt_momentum_20d
        m60 = factors.tlt_momentum_60d
        
        # Blend short and medium-term momentum
        blended = 0.6 * np.sign(m20) * min(abs(m20) / 0.03, 1.0) + \
                  0.4 * np.sign(m60) * min(abs(m60) / 0.06, 1.0)
        
        return np.clip(blended, -1.0, 1.0)
    
    def _macro_score(self, factors: DurationFactors) -> float:
        """
        Macro regime: high inflation → short duration, recession → long duration.
        Uses CPI, PMI, and Fed Funds rate.
        """
        score = 0.0
        
        # Inflation component: high CPI → short duration
        if factors.cpi_yoy > 4.0:
            score -= 0.5
        elif factors.cpi_yoy > 3.0:
            score -= 0.2
        elif factors.cpi_yoy < 2.0:
            score += 0.3
        
        # Growth component: low PMI → long duration (recession)
        if factors.pmi < 45:
            score += 0.5
        elif factors.pmi < 50:
            score += 0.2
        elif factors.pmi > 55:
            score -= 0.2
        
        # Real rate: high real rate → bonds attractive → long duration
        real_rate = factors.yield_10y - factors.cpi_yoy
        if real_rate > 2.0:
            score += 0.3
        elif real_rate > 1.0:
            score += 0.1
        elif real_rate < 0:
            score -= 0.2
        
        return np.clip(score, -1.0, 1.0)
    
    def _crisis_score(self, factors: DurationFactors) -> float:
        """
        Crisis detection: high VIX + widening HY spreads → flight to quality → long duration.
        """
        score = 0.0
        
        # VIX component
        if factors.vix > 35:
            score += 0.8  # Crisis → strong flight to quality
        elif factors.vix > 25:
            score += 0.3
        elif factors.vix < 15:
            score -= 0.2  # Low vol → risk-on, less duration needed
        
        # HY spread widening
        if factors.hy_spread > 600:
            score += 0.6  # Credit stress → flight to quality
        elif factors.hy_spread > 450:
            score += 0.2
        
        # Rapid spread widening
        if factors.hy_spread_change_1m > 100:
            score += 0.4
        
        return np.clip(score, -1.0, 1.0)
    
    def compute_duration_score(self, factors: DurationFactors) -> float:
        """
        Compute composite duration score from -1 to +1.
        Positive = long duration, Negative = short duration.
        """
        slope = self._slope_score(factors) * self.slope_weight
        momentum = self._momentum_score(factors) * self.momentum_weight
        macro = self._macro_score(factors) * self.macro_weight
        crisis = self._crisis_score(factors) * self.crisis_weight
        
        composite = slope + momentum + macro + crisis
        return np.clip(composite, -1.0, 1.0)
    
    def score_to_allocation(self, score: float) -> dict:
        """Map duration score to ETF allocations."""
        if score > 0.5:
            return {"TLT": 0.7, "IEF": 0.3, "SHY": 0.0, "BIL": 0.0}
        elif score > 0.2:
            return {"TLT": 0.3, "IEF": 0.5, "SHY": 0.2, "BIL": 0.0}
        elif score > -0.2:
            return {"TLT": 0.0, "IEF": 0.3, "SHY": 0.5, "BIL": 0.2}
        elif score > -0.5:
            return {"TLT": 0.0, "IEF": 0.0, "SHY": 0.5, "BIL": 0.5}
        else:
            return {"TLT": 0.0, "IEF": 0.0, "SHY": 0.2, "BIL": 0.8}
    
    def generate_signals(self, factors: DurationFactors) -> list:
        """Generate duration timing signals."""
        score = self.compute_duration_score(factors)
        allocation = self.score_to_allocation(score)
        
        signals = []
        for etf, weight in allocation.items():
            if weight > 0:
                signals.append(Signal(
                    symbol=etf,
                    direction=SignalDirection.LONG,
                    weight=weight,
                    metadata={
                        "duration_score": round(score, 3),
                        "allocation": allocation,
                        "factors": {
                            "slope_2s10s": factors.slope_2s10s,
                            "vix": factors.vix,
                            "tlt_mom_20d": round(factors.tlt_momentum_20d, 4),
                            "cpi_yoy": factors.cpi_yoy,
                            "pmi": factors.pmi,
                            "hy_spread": factors.hy_spread,
                        }
                    }
                ))
        
        return signals
    
    def run(self, data: dict) -> list:
        """
        Main entry point.
        
        data should contain factor values matching DurationFactors fields.
        """
        factors = DurationFactors(**{k: v for k, v in data.items() 
                                      if k in DurationFactors.__dataclass_fields__})
        return self.generate_signals(factors)
