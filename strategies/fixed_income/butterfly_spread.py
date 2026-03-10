"""
FI_BFLY_003 — Treasury Butterfly Spread Strategy
==================================================
Trades curvature of the yield curve using butterfly spreads.
A butterfly is long the wings (short + long tenor) and short the belly (mid tenor),
or vice versa, to profit from changes in curve curvature.

Key structures:
- 2s5s10s butterfly: classic belly trade
- 5s10s30s butterfly: long-end curvature
- 2s10s30s butterfly: barbell vs bullet
- Weighted butterflies: duration-neutral construction

Signals based on:
- Z-score of curvature relative to history
- Carry analysis (which side earns positive carry)
- Regime-conditional entry (different thresholds for different macro regimes)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import numpy as np

from strategies.base import StrategyBase, Signal, SignalDirection, AssetClass, StrategyStyle


@dataclass
class ButterflyConfig:
    """Configuration for a specific butterfly structure."""
    name: str
    short_tenor: str
    belly_tenor: str
    long_tenor: str
    short_maturity: float  # years
    belly_maturity: float
    long_maturity: float
    short_dv01_weight: float  # duration-neutral weights
    belly_dv01_weight: float
    long_dv01_weight: float


# Standard butterfly structures
BUTTERFLY_CONFIGS = {
    "2s5s10s": ButterflyConfig(
        name="2s5s10s", short_tenor="2Y", belly_tenor="5Y", long_tenor="10Y",
        short_maturity=2.0, belly_maturity=5.0, long_maturity=10.0,
        short_dv01_weight=0.375, belly_dv01_weight=1.0, long_dv01_weight=0.625,
    ),
    "5s10s30s": ButterflyConfig(
        name="5s10s30s", short_tenor="5Y", belly_tenor="10Y", long_tenor="30Y",
        short_maturity=5.0, belly_maturity=10.0, long_maturity=30.0,
        short_dv01_weight=0.333, belly_dv01_weight=1.0, long_dv01_weight=0.667,
    ),
    "2s10s30s": ButterflyConfig(
        name="2s10s30s", short_tenor="2Y", belly_tenor="10Y", long_tenor="30Y",
        short_maturity=2.0, belly_maturity=10.0, long_maturity=30.0,
        short_dv01_weight=0.286, belly_dv01_weight=1.0, long_dv01_weight=0.714,
    ),
}


class ButterflySpreadStrategy(StrategyBase):
    """
    Treasury Butterfly Spread Strategy.
    
    Trades curvature using duration-weighted butterfly spreads.
    Positive butterfly = long wings, short belly (profit from curve flattening at belly).
    Negative butterfly = short wings, long belly (profit from curve steepening at belly).
    """
    
    def __init__(self, config: Optional[dict] = None):
        super().__init__(
            strategy_id="FI_BFLY_003",
            name="Treasury Butterfly Spread",
            asset_class=AssetClass.FIXED_INCOME,
            style=StrategyStyle.RELATIVE_VALUE,
            description="Trades yield curve curvature using duration-weighted butterfly spreads.",
        )
        self.config = config or {}
        self.z_entry = self.config.get("z_entry", 1.5)
        self.z_exit = self.config.get("z_exit", 0.5)
        self.lookback = self.config.get("lookback_days", 252)
        self.min_history = self.config.get("min_history", 60)
        self.max_weight = self.config.get("max_weight", 0.15)
        self.structures = self.config.get("structures", list(BUTTERFLY_CONFIGS.keys()))
    
    def required_data(self):
        return {"yields": "FRED:DGS*", "curvature_histories": "computed:butterfly_history"}

    def compute_curvature(self, yields: dict, bfly_config: ButterflyConfig) -> Optional[float]:
        """
        Compute butterfly spread (curvature).
        Butterfly = 2 * belly_yield - short_yield - long_yield
        """
        short_y = yields.get(bfly_config.short_tenor)
        belly_y = yields.get(bfly_config.belly_tenor)
        long_y = yields.get(bfly_config.long_tenor)
        
        if any(v is None for v in [short_y, belly_y, long_y]):
            return None
        
        return 2 * belly_y - short_y - long_y
    
    def compute_carry(self, yields: dict, bfly_config: ButterflyConfig) -> float:
        """
        Estimate carry for the butterfly position.
        Positive carry = belly yields more than wings (sell belly is positive carry).
        """
        short_y = yields.get(bfly_config.short_tenor, 0)
        belly_y = yields.get(bfly_config.belly_tenor, 0)
        long_y = yields.get(bfly_config.long_tenor, 0)
        
        # Carry for selling belly: belly yield - weighted average of wings
        wing_avg = (short_y * bfly_config.short_dv01_weight + 
                    long_y * bfly_config.long_dv01_weight) / \
                   (bfly_config.short_dv01_weight + bfly_config.long_dv01_weight)
        
        return belly_y - wing_avg
    
    def generate_signals(self, yields: dict, 
                         curvature_histories: dict = None) -> list:
        """
        Generate butterfly trading signals.
        
        Args:
            yields: dict of tenor -> yield (e.g., {"2Y": 4.5, "5Y": 4.2, ...})
            curvature_histories: dict of structure_name -> list of historical curvature values
        """
        signals = []
        curvature_histories = curvature_histories or {}
        
        for struct_name in self.structures:
            if struct_name not in BUTTERFLY_CONFIGS:
                continue
            
            bfly = BUTTERFLY_CONFIGS[struct_name]
            curvature = self.compute_curvature(yields, bfly)
            
            if curvature is None:
                continue
            
            carry = self.compute_carry(yields, bfly)
            history = curvature_histories.get(struct_name, [])
            
            # Z-score based signal
            if len(history) >= self.min_history:
                arr = np.array(history[-self.lookback:])
                mean = np.mean(arr)
                std = np.std(arr)
                
                if std < 0.001:
                    continue
                
                z = (curvature - mean) / std
                
                if z > self.z_entry:
                    # Curvature is high → sell belly (expect curvature to decrease)
                    # This is a "positive butterfly" — long wings, short belly
                    weight = min(abs(z) / 5.0, self.max_weight)
                    
                    # Carry adjustment: boost if carry is favorable
                    if carry > 0:
                        weight = min(weight * 1.2, self.max_weight)
                    
                    signals.append(Signal(
                        symbol=f"{struct_name}_SELL_BELLY",
                        direction=SignalDirection.SHORT,
                        weight=weight,
                        metadata={
                            "structure": struct_name,
                            "trade": "positive_butterfly",
                            "curvature": round(curvature, 4),
                            "z_score": round(z, 2),
                            "carry_bps": round(carry * 100, 1),
                            "action": f"Long {bfly.short_tenor}+{bfly.long_tenor}, "
                                      f"Short {bfly.belly_tenor}",
                            "weights": {
                                bfly.short_tenor: bfly.short_dv01_weight,
                                bfly.belly_tenor: -bfly.belly_dv01_weight,
                                bfly.long_tenor: bfly.long_dv01_weight,
                            }
                        }
                    ))
                
                elif z < -self.z_entry:
                    # Curvature is low → buy belly (expect curvature to increase)
                    # This is a "negative butterfly" — short wings, long belly
                    weight = min(abs(z) / 5.0, self.max_weight)
                    
                    if carry < 0:
                        weight = min(weight * 1.2, self.max_weight)
                    
                    signals.append(Signal(
                        symbol=f"{struct_name}_BUY_BELLY",
                        direction=SignalDirection.LONG,
                        weight=weight,
                        metadata={
                            "structure": struct_name,
                            "trade": "negative_butterfly",
                            "curvature": round(curvature, 4),
                            "z_score": round(z, 2),
                            "carry_bps": round(carry * 100, 1),
                            "action": f"Short {bfly.short_tenor}+{bfly.long_tenor}, "
                                      f"Long {bfly.belly_tenor}",
                            "weights": {
                                bfly.short_tenor: -bfly.short_dv01_weight,
                                bfly.belly_tenor: bfly.belly_dv01_weight,
                                bfly.long_tenor: -bfly.long_dv01_weight,
                            }
                        }
                    ))
        
        return signals
    
    def run(self, data: dict) -> list:
        """
        Main entry point.
        
        data should contain:
        - yields: dict of tenor -> yield
        - curvature_histories: dict of structure_name -> list of historical values
        """
        yields = data.get("yields", {})
        if not yields:
            return []
        
        return self.generate_signals(
            yields,
            curvature_histories=data.get("curvature_histories", {}),
        )
