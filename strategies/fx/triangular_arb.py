"""
FX_TARB_003 — Triangular Arbitrage Strategy
=============================================
Detects and exploits triangular arbitrage opportunities in FX markets.
Given three currency pairs forming a triangle (e.g., EUR/USD, USD/JPY,
EUR/JPY), the strategy identifies when the cross rate deviates from
the implied rate, creating a risk-free profit opportunity.

Key features:
- Scans all possible G10 triangles (120 combinations)
- Computes implied cross rates and deviations
- Filters by minimum profit threshold (net of transaction costs)
- Latency-aware: flags opportunities but acknowledges execution risk
- Tracks historical deviation patterns for statistical edge
- Can also be used as a relative value signal (slower mean reversion)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from itertools import combinations
import numpy as np

from strategies.base import StrategyBase, Signal, SignalDirection, AssetClass, StrategyStyle


@dataclass
class FXQuote:
    pair: str          # e.g., "EURUSD"
    base: str          # e.g., "EUR"
    quote: str         # e.g., "USD"
    bid: float
    ask: float
    mid: float = 0.0
    spread_bps: float = 0.0
    
    def __post_init__(self):
        if self.mid == 0:
            self.mid = (self.bid + self.ask) / 2
        if self.spread_bps == 0 and self.mid > 0:
            self.spread_bps = (self.ask - self.bid) / self.mid * 10000


@dataclass
class TriangleOpportunity:
    currencies: tuple  # (A, B, C)
    path: str          # "A→B→C→A"
    implied_rate: float
    market_rate: float
    deviation_bps: float
    profit_bps: float  # after transaction costs
    legs: list         # list of (pair, direction, rate)


class TriangularArbStrategy(StrategyBase):
    """
    Triangular Arbitrage Strategy.
    
    Scans G10 FX pairs for triangular arbitrage opportunities where
    the cross rate deviates from the implied rate by more than
    transaction costs.
    """
    
    def __init__(self, config: Optional[dict] = None):
        super().__init__(
            strategy_id="FX_TARB_003",
            name="Triangular Arbitrage",
            asset_class=AssetClass.FX,
            style=StrategyStyle.STATISTICAL_ARB,
            description="Detects triangular arbitrage opportunities in G10 FX.",
        )
        self.config = config or {}
        self.min_profit_bps = self.config.get("min_profit_bps", 2.0)
        self.transaction_cost_bps = self.config.get("transaction_cost_bps", 1.5)
        self.max_weight = self.config.get("max_weight", 0.10)
        self.use_statistical_mode = self.config.get("statistical_mode", True)
        self.stat_z_threshold = self.config.get("stat_z_threshold", 2.0)
    
    def required_data(self):
        return {"quotes": "LIVE:G10_FX_QUOTES", "deviation_histories": "computed:triangle_deviations"}

    def get_rate(self, quotes: dict, base: str, quote: str) -> Optional[tuple]:
        """
        Get the exchange rate for base/quote from available quotes.
        Returns (rate, direction) where direction is 'direct' or 'inverse'.
        """
        direct_pair = f"{base}{quote}"
        inverse_pair = f"{quote}{base}"
        
        if direct_pair in quotes:
            q = quotes[direct_pair]
            return (q.mid, "direct", q.spread_bps)
        elif inverse_pair in quotes:
            q = quotes[inverse_pair]
            if q.mid > 0:
                return (1.0 / q.mid, "inverse", q.spread_bps)
        
        return None
    
    def find_triangles(self, quotes: dict) -> list:
        """
        Find all valid currency triangles from available quotes.
        A triangle (A, B, C) requires rates for A/B, B/C, and A/C.
        """
        # Extract all currencies from quotes
        currencies = set()
        for pair_name, quote in quotes.items():
            currencies.add(quote.base)
            currencies.add(quote.quote)
        
        triangles = []
        for combo in combinations(sorted(currencies), 3):
            a, b, c = combo
            
            # Check if all three legs exist
            rate_ab = self.get_rate(quotes, a, b)
            rate_bc = self.get_rate(quotes, b, c)
            rate_ac = self.get_rate(quotes, a, c)
            
            if all(r is not None for r in [rate_ab, rate_bc, rate_ac]):
                triangles.append((a, b, c, rate_ab, rate_bc, rate_ac))
        
        return triangles
    
    def compute_triangle_deviation(self, rate_ab: float, rate_bc: float,
                                    rate_ac: float) -> float:
        """
        Compute the deviation in a triangle A→B→C→A.
        
        If we start with 1 unit of A:
        1. Convert A→B: get rate_ab units of B
        2. Convert B→C: get rate_ab * rate_bc units of C
        3. Convert C→A: get rate_ab * rate_bc / rate_ac units of A
        
        Profit = result - 1.0
        """
        if rate_ac == 0:
            return 0.0
        
        result = rate_ab * rate_bc / rate_ac
        return (result - 1.0) * 10000  # in basis points
    
    def scan_opportunities(self, quotes: dict) -> list:
        """Scan all triangles for arbitrage opportunities."""
        triangles = self.find_triangles(quotes)
        opportunities = []
        
        for a, b, c, (rate_ab, dir_ab, spread_ab), \
                       (rate_bc, dir_bc, spread_bc), \
                       (rate_ac, dir_ac, spread_ac) in triangles:
            
            # Forward path: A→B→C→A
            dev_forward = self.compute_triangle_deviation(rate_ab, rate_bc, rate_ac)
            total_spread = spread_ab + spread_bc + spread_ac
            profit_forward = abs(dev_forward) - total_spread - self.transaction_cost_bps * 3
            
            if abs(dev_forward) > self.min_profit_bps:
                opportunities.append(TriangleOpportunity(
                    currencies=(a, b, c),
                    path=f"{a}→{b}→{c}→{a}" if dev_forward > 0 else f"{a}→{c}→{b}→{a}",
                    implied_rate=rate_ab * rate_bc,
                    market_rate=rate_ac,
                    deviation_bps=round(dev_forward, 2),
                    profit_bps=round(profit_forward, 2),
                    legs=[
                        (f"{a}{b}", "buy" if dev_forward > 0 else "sell", rate_ab),
                        (f"{b}{c}", "buy" if dev_forward > 0 else "sell", rate_bc),
                        (f"{a}{c}", "sell" if dev_forward > 0 else "buy", rate_ac),
                    ]
                ))
        
        # Sort by absolute deviation
        opportunities.sort(key=lambda x: abs(x.deviation_bps), reverse=True)
        return opportunities
    
    def generate_signals(self, quotes: dict,
                         deviation_histories: dict = None) -> list:
        """
        Generate arbitrage/relative value signals.
        
        Args:
            quotes: dict of pair_name -> FXQuote
            deviation_histories: dict of triangle_key -> list of historical deviations
        """
        opportunities = self.scan_opportunities(quotes)
        deviation_histories = deviation_histories or {}
        signals = []
        
        for opp in opportunities:
            key = "_".join(opp.currencies)
            
            # Pure arbitrage signal (if profit > 0 after costs)
            if opp.profit_bps > 0:
                signals.append(Signal(
                    symbol=f"TRIANGLE_{key}",
                    direction=SignalDirection.LONG,
                    weight=min(opp.profit_bps / 50.0, self.max_weight),
                    metadata={
                        "type": "pure_arbitrage",
                        "path": opp.path,
                        "deviation_bps": opp.deviation_bps,
                        "profit_bps": opp.profit_bps,
                        "legs": [(l[0], l[1], round(l[2], 6)) for l in opp.legs],
                        "warning": "Execution risk: latency may eliminate profit",
                    }
                ))
            
            # Statistical relative value signal
            if self.use_statistical_mode and key in deviation_histories:
                history = deviation_histories[key]
                if len(history) >= 60:
                    arr = np.array(history[-252:])
                    z = (opp.deviation_bps - np.mean(arr)) / max(np.std(arr), 0.1)
                    
                    if abs(z) > self.stat_z_threshold:
                        direction = SignalDirection.SHORT if z > 0 else SignalDirection.LONG
                        signals.append(Signal(
                            symbol=f"TRIANGLE_RV_{key}",
                            direction=direction,
                            weight=min(abs(z) / 6.0, self.max_weight),
                            metadata={
                                "type": "statistical_rv",
                                "path": opp.path,
                                "deviation_bps": opp.deviation_bps,
                                "z_score": round(z, 2),
                                "mean_deviation": round(float(np.mean(arr)), 2),
                                "action": "Mean reversion on cross-rate deviation",
                            }
                        ))
        
        return signals
    
    def run(self, data: dict) -> list:
        """
        Main entry point.
        
        data should contain:
        - quotes: dict of pair_name -> {bid, ask, base, quote}
        - deviation_histories: dict of triangle_key -> list (optional)
        """
        raw_quotes = data.get("quotes", {})
        quotes = {}
        for pair_name, q in raw_quotes.items():
            if isinstance(q, dict):
                quotes[pair_name] = FXQuote(
                    pair=pair_name,
                    base=q.get("base", pair_name[:3]),
                    quote=q.get("quote", pair_name[3:]),
                    bid=q.get("bid", 0),
                    ask=q.get("ask", 0),
                )
            elif isinstance(q, FXQuote):
                quotes[pair_name] = q
        
        return self.generate_signals(
            quotes,
            deviation_histories=data.get("deviation_histories", {}),
        )
