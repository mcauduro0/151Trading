"""
FX_CARRY_001 — G10 FX Carry Trade Strategy
=============================================
Classic carry trade: go long high-yielding currencies, short low-yielding
currencies in the G10 universe. Uses forward points / interest rate
differentials as the carry signal, with risk overlays for crash protection.

Key features:
- G10 universe: USD, EUR, GBP, JPY, CHF, AUD, NZD, CAD, NOK, SEK
- Carry signal from 3-month interest rate differentials
- Momentum overlay (3M FX returns) to avoid catching falling knives
- VIX-based crash protection (reduce exposure when VIX > 25)
- Cross-sectional ranking: long top 3, short bottom 3
- Position sizing based on carry-to-vol ratio
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import numpy as np

from strategies.base import StrategyBase, Signal, SignalDirection, AssetClass, StrategyStyle


# G10 currencies and their typical interest rate FRED proxies
G10_CURRENCIES = {
    "USD": {"name": "US Dollar", "rate_proxy": "DFF"},
    "EUR": {"name": "Euro", "rate_proxy": "ECBDFR"},
    "GBP": {"name": "British Pound", "rate_proxy": "IUDSOIA"},
    "JPY": {"name": "Japanese Yen", "rate_proxy": "IRSTCI01JPM156N"},
    "CHF": {"name": "Swiss Franc", "rate_proxy": "IRSTCI01CHM156N"},
    "AUD": {"name": "Australian Dollar", "rate_proxy": "IRSTCI01AUM156N"},
    "NZD": {"name": "New Zealand Dollar", "rate_proxy": "IRSTCI01NZM156N"},
    "CAD": {"name": "Canadian Dollar", "rate_proxy": "IRSTCI01CAM156N"},
    "NOK": {"name": "Norwegian Krone", "rate_proxy": "IRSTCI01NOM156N"},
    "SEK": {"name": "Swedish Krona", "rate_proxy": "IRSTCI01SEM156N"},
}

# FX pairs (all vs USD)
FX_PAIRS = {
    "EUR": "EURUSD", "GBP": "GBPUSD", "JPY": "USDJPY",
    "CHF": "USDCHF", "AUD": "AUDUSD", "NZD": "NZDUSD",
    "CAD": "USDCAD", "NOK": "USDNOK", "SEK": "USDSEK",
}


@dataclass
class CurrencyCarryData:
    currency: str
    short_rate: float  # 3-month rate in %
    fx_return_3m: float = 0.0  # 3-month FX return vs USD
    fx_return_1m: float = 0.0
    fx_vol_3m: float = 0.10  # annualized vol
    carry_vs_usd: float = 0.0  # rate differential vs USD


class G10CarryStrategy(StrategyBase):
    """
    G10 FX Carry Trade Strategy.
    
    Ranks currencies by carry (interest rate differential vs USD),
    applies momentum and volatility filters, and constructs a
    long-short portfolio of the top/bottom 3 currencies.
    """
    
    def __init__(self, config: Optional[dict] = None):
        super().__init__(
            strategy_id="FX_CARRY_001",
            name="G10 FX Carry Trade",
            asset_class=AssetClass.FX,
            style=StrategyStyle.CARRY,
            description="Classic G10 carry trade: long high-yield, short low-yield currencies.",
        )
        self.config = config or {}
        self.n_long = self.config.get("n_long", 3)
        self.n_short = self.config.get("n_short", 3)
        self.carry_weight = self.config.get("carry_weight", 0.6)
        self.momentum_weight = self.config.get("momentum_weight", 0.3)
        self.vol_weight = self.config.get("vol_weight", 0.1)
        self.vix_threshold = self.config.get("vix_threshold", 25)
        self.vix_scale_factor = self.config.get("vix_scale_factor", 0.5)
        self.max_position_weight = self.config.get("max_position_weight", 0.20)
    
    def required_data(self):
        return {"rates": "FRED:G10_SHORT_RATES", "fx": "YAHOO:G10_FX_PAIRS", "vix": "FRED:VIXCLS"}

    def compute_carry_scores(self, currencies: list, usd_rate: float) -> list:
        """Compute carry score for each currency relative to USD."""
        for ccy in currencies:
            ccy.carry_vs_usd = ccy.short_rate - usd_rate
        return currencies
    
    def compute_composite_score(self, ccy: CurrencyCarryData) -> float:
        """
        Composite score blending carry, momentum, and inverse volatility.
        Higher score = more attractive to go long.
        """
        # Normalize carry (typical range -3% to +3%)
        carry_norm = ccy.carry_vs_usd / 3.0
        
        # Normalize momentum (typical range -10% to +10%)
        mom_norm = ccy.fx_return_3m / 0.10
        
        # Inverse vol (lower vol = better carry-to-vol ratio)
        vol_norm = -ccy.fx_vol_3m / 0.15  # negative because lower vol is better
        
        score = (self.carry_weight * carry_norm +
                 self.momentum_weight * mom_norm +
                 self.vol_weight * vol_norm)
        
        return score
    
    def generate_signals(self, currencies: list, usd_rate: float,
                         vix: float = 20.0) -> list:
        """
        Generate carry trade signals.
        
        Args:
            currencies: list of CurrencyCarryData for G10 (excluding USD)
            usd_rate: current USD short-term rate
            vix: current VIX level for crash protection
        """
        if len(currencies) < self.n_long + self.n_short:
            return []
        
        # Compute carry vs USD
        currencies = self.compute_carry_scores(currencies, usd_rate)
        
        # Compute composite scores
        scored = [(ccy, self.compute_composite_score(ccy)) for ccy in currencies]
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # VIX crash protection
        scale = 1.0
        if vix > self.vix_threshold:
            scale = self.vix_scale_factor
            if vix > 35:
                scale = 0.25  # Severe risk-off
        
        signals = []
        
        # Long top N (highest carry + momentum)
        for ccy, score in scored[:self.n_long]:
            pair = FX_PAIRS.get(ccy.currency, f"{ccy.currency}USD")
            weight = min(self.max_position_weight * scale, 0.20)
            
            signals.append(Signal(
                symbol=pair,
                direction=SignalDirection.LONG,
                weight=weight,
                metadata={
                    "currency": ccy.currency,
                    "carry_vs_usd": round(ccy.carry_vs_usd, 3),
                    "composite_score": round(score, 3),
                    "fx_return_3m": round(ccy.fx_return_3m, 4),
                    "fx_vol_3m": round(ccy.fx_vol_3m, 4),
                    "vix_scale": scale,
                    "side": "long_carry",
                }
            ))
        
        # Short bottom N (lowest carry + momentum)
        for ccy, score in scored[-self.n_short:]:
            pair = FX_PAIRS.get(ccy.currency, f"{ccy.currency}USD")
            weight = min(self.max_position_weight * scale, 0.20)
            
            signals.append(Signal(
                symbol=pair,
                direction=SignalDirection.SHORT,
                weight=weight,
                metadata={
                    "currency": ccy.currency,
                    "carry_vs_usd": round(ccy.carry_vs_usd, 3),
                    "composite_score": round(score, 3),
                    "fx_return_3m": round(ccy.fx_return_3m, 4),
                    "fx_vol_3m": round(ccy.fx_vol_3m, 4),
                    "vix_scale": scale,
                    "side": "short_carry",
                }
            ))
        
        return signals
    
    def run(self, data: dict) -> list:
        """
        Main entry point.
        
        data should contain:
        - currencies: list of dicts with currency, short_rate, fx_return_3m, fx_vol_3m
        - usd_rate: float
        - vix: float (optional, default 20)
        """
        raw_currencies = data.get("currencies", [])
        usd_rate = data.get("usd_rate", 5.0)
        vix = data.get("vix", 20.0)
        
        currencies = [
            CurrencyCarryData(**{k: v for k, v in c.items() 
                                 if k in CurrencyCarryData.__dataclass_fields__})
            for c in raw_currencies
        ]
        
        return self.generate_signals(currencies, usd_rate, vix)
