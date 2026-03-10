"""
FX_PPP_002 — Purchasing Power Parity (PPP) Value Strategy
==========================================================
Long-term FX value strategy based on deviations from Purchasing Power
Parity. Currencies trading significantly below their PPP-implied fair
value are expected to appreciate over time (mean reversion).

Key features:
- PPP fair value estimation using Big Mac Index / OECD PPP data
- Real Effective Exchange Rate (REER) z-score analysis
- Multi-horizon valuation (short-term vs structural misalignment)
- Momentum filter to avoid value traps
- Combination with carry for enhanced signal quality
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import numpy as np

from strategies.base import StrategyBase, Signal, SignalDirection, AssetClass, StrategyStyle


@dataclass
class PPPCurrencyData:
    currency: str
    spot_rate: float          # current spot rate vs USD
    ppp_fair_value: float     # PPP-implied fair value vs USD
    reer_index: float = 100.0 # Real Effective Exchange Rate index
    reer_5y_avg: float = 100.0
    fx_return_6m: float = 0.0 # 6-month momentum
    fx_return_12m: float = 0.0
    carry_differential: float = 0.0  # interest rate diff vs USD


class PPPValueStrategy(StrategyBase):
    """
    PPP Value Strategy for G10 currencies.
    
    Identifies currencies that are significantly over/undervalued
    relative to PPP fair value and generates mean-reversion signals.
    Uses REER z-scores and momentum filters.
    """
    
    def __init__(self, config: Optional[dict] = None):
        super().__init__(
            strategy_id="FX_PPP_002",
            name="PPP Value (Purchasing Power Parity)",
            asset_class=AssetClass.FX,
            style=StrategyStyle.VALUE,
            description="Long-term FX value strategy based on PPP deviations.",
        )
        self.config = config or {}
        self.misalignment_threshold = self.config.get("misalignment_threshold", 0.15)
        self.strong_misalignment = self.config.get("strong_misalignment", 0.30)
        self.momentum_filter = self.config.get("momentum_filter", True)
        self.reer_zscore_threshold = self.config.get("reer_zscore_threshold", 1.0)
        self.max_weight = self.config.get("max_weight", 0.15)
        self.n_positions = self.config.get("n_positions", 3)
    
    def required_data(self):
        return {"ppp": "OECD:PPP_DATA", "reer": "BIS:REER", "fx": "YAHOO:G10_FX_PAIRS"}

    def compute_misalignment(self, ccy: PPPCurrencyData) -> float:
        """
        Compute PPP misalignment as percentage deviation from fair value.
        Positive = overvalued (spot stronger than PPP), Negative = undervalued.
        
        For pairs quoted as XXXUSD (e.g., EURUSD):
            misalignment = (spot - fair_value) / fair_value
            Positive = XXX overvalued vs USD
        """
        if ccy.ppp_fair_value <= 0:
            return 0.0
        return (ccy.spot_rate - ccy.ppp_fair_value) / ccy.ppp_fair_value
    
    def compute_reer_zscore(self, ccy: PPPCurrencyData) -> float:
        """
        Compute REER z-score relative to 5-year average.
        High REER = currency is expensive in real terms.
        """
        if ccy.reer_5y_avg <= 0:
            return 0.0
        return (ccy.reer_index - ccy.reer_5y_avg) / max(ccy.reer_5y_avg * 0.05, 1.0)
    
    def compute_value_score(self, ccy: PPPCurrencyData) -> float:
        """
        Composite value score. More negative = more undervalued = better buy.
        """
        misalignment = self.compute_misalignment(ccy)
        reer_z = self.compute_reer_zscore(ccy)
        
        # Blend PPP misalignment and REER z-score
        value_score = -0.6 * misalignment + -0.4 * (reer_z / 3.0)
        
        # Momentum filter: penalize if momentum is against value
        if self.momentum_filter:
            if value_score > 0 and ccy.fx_return_6m < -0.05:
                # Undervalued but still falling → reduce conviction
                value_score *= 0.5
            elif value_score < 0 and ccy.fx_return_6m > 0.05:
                # Overvalued but still rising → reduce conviction
                value_score *= 0.5
        
        return value_score
    
    def generate_signals(self, currencies: list) -> list:
        """
        Generate PPP value signals.
        
        Args:
            currencies: list of PPPCurrencyData
        """
        if len(currencies) < 2:
            return []
        
        # Score all currencies
        scored = []
        for ccy in currencies:
            misalignment = self.compute_misalignment(ccy)
            value_score = self.compute_value_score(ccy)
            scored.append((ccy, value_score, misalignment))
        
        # Sort by value score (most undervalued first)
        scored.sort(key=lambda x: x[1], reverse=True)
        
        signals = []
        
        # Long the most undervalued
        for ccy, score, misalignment in scored[:self.n_positions]:
            if abs(misalignment) < self.misalignment_threshold:
                continue  # Not enough misalignment
            
            if misalignment < 0:  # Undervalued → buy
                weight = min(abs(misalignment) / 0.50, self.max_weight)
                
                if abs(misalignment) > self.strong_misalignment:
                    weight = min(weight * 1.3, self.max_weight)
                
                signals.append(Signal(
                    symbol=f"{ccy.currency}USD",
                    direction=SignalDirection.LONG,
                    weight=weight,
                    metadata={
                        "currency": ccy.currency,
                        "ppp_misalignment": round(misalignment, 4),
                        "value_score": round(score, 3),
                        "reer_index": ccy.reer_index,
                        "reer_zscore": round(self.compute_reer_zscore(ccy), 2),
                        "fx_return_6m": round(ccy.fx_return_6m, 4),
                        "side": "long_undervalued",
                    }
                ))
        
        # Short the most overvalued
        for ccy, score, misalignment in scored[-self.n_positions:]:
            if abs(misalignment) < self.misalignment_threshold:
                continue
            
            if misalignment > 0:  # Overvalued → sell
                weight = min(abs(misalignment) / 0.50, self.max_weight)
                
                if abs(misalignment) > self.strong_misalignment:
                    weight = min(weight * 1.3, self.max_weight)
                
                signals.append(Signal(
                    symbol=f"{ccy.currency}USD",
                    direction=SignalDirection.SHORT,
                    weight=weight,
                    metadata={
                        "currency": ccy.currency,
                        "ppp_misalignment": round(misalignment, 4),
                        "value_score": round(score, 3),
                        "reer_index": ccy.reer_index,
                        "reer_zscore": round(self.compute_reer_zscore(ccy), 2),
                        "fx_return_6m": round(ccy.fx_return_6m, 4),
                        "side": "short_overvalued",
                    }
                ))
        
        return signals
    
    def run(self, data: dict) -> list:
        """
        Main entry point.
        
        data should contain:
        - currencies: list of dicts with currency, spot_rate, ppp_fair_value, etc.
        """
        raw = data.get("currencies", [])
        currencies = [
            PPPCurrencyData(**{k: v for k, v in c.items()
                               if k in PPPCurrencyData.__dataclass_fields__})
            for c in raw
        ]
        return self.generate_signals(currencies)
