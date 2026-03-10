"""
151 Trading System — Options Structure Analyzer
=================================================
Analyzes multi-leg option structures: payoff diagrams, risk metrics,
scenario analysis, and structure recommendations.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from strategies.options.engine.pricing import (
    OptionLeg, OptionType, PositionSide, Greeks,
    bs_price, bs_greeks, structure_payoff_at_expiry,
    structure_greeks, max_profit, max_loss,
    breakeven_points, probability_of_profit,
)


@dataclass
class PayoffDiagram:
    """Payoff diagram data for charting."""
    spot_range: List[float]
    pnl_at_expiry: List[float]
    breakevens: List[float]
    max_profit: float
    max_loss: float
    current_spot: float


@dataclass
class StructureAnalysis:
    """Complete analysis of an option structure."""
    name: str
    legs: List[OptionLeg]
    net_premium: float          # net debit (positive) or credit (negative)
    greeks: Greeks
    payoff: PayoffDiagram
    pop: float                  # probability of profit
    risk_reward_ratio: float    # |max_profit / max_loss|
    margin_requirement: float   # estimated margin
    max_profit_price: float     # spot price at max profit
    max_loss_price: float       # spot price at max loss


class StructureAnalyzer:
    """Analyze option structures for risk, reward, and Greeks."""

    def __init__(self, risk_free_rate: float = 0.05):
        self.r = risk_free_rate

    def analyze(self, legs: List[OptionLeg], S: float,
                name: str = "Custom Structure",
                sigma: float = 0.20) -> StructureAnalysis:
        """
        Full analysis of a multi-leg option structure.

        Args:
            legs: List of OptionLeg objects
            S: Current spot price
            name: Structure name
            sigma: Volatility for PoP calculation
        """
        # Spot range: +/- 30% from current price
        spot_min = S * 0.70
        spot_max = S * 1.30
        spot_range = np.linspace(spot_min, spot_max, 500)

        # Payoff at expiry
        pnl = structure_payoff_at_expiry(legs, spot_range)

        # Key metrics
        mp = float(np.max(pnl))
        ml = float(np.min(pnl))
        be = breakeven_points(legs, spot_range)

        # Max profit/loss spot prices
        mp_idx = int(np.argmax(pnl))
        ml_idx = int(np.argmin(pnl))
        mp_price = float(spot_range[mp_idx])
        ml_price = float(spot_range[ml_idx])

        # Net premium
        net_prem = sum(
            leg.premium * leg.side.value * leg.quantity
            for leg in legs
        )

        # Greeks
        greeks = structure_greeks(legs, S, self.r)

        # Probability of profit
        avg_dte = np.mean([leg.expiry_days for leg in legs])
        T = avg_dte / 365.0
        pop = probability_of_profit(legs, S, sigma, T, self.r)

        # Risk-reward ratio
        rr = abs(mp / ml) if ml != 0 else float('inf')

        # Margin estimate (simplified: max loss * 1.2)
        margin = abs(ml) * 1.2

        payoff = PayoffDiagram(
            spot_range=spot_range.tolist(),
            pnl_at_expiry=pnl.tolist(),
            breakevens=be,
            max_profit=mp,
            max_loss=ml,
            current_spot=S,
        )

        return StructureAnalysis(
            name=name,
            legs=legs,
            net_premium=net_prem,
            greeks=greeks,
            payoff=payoff,
            pop=pop,
            risk_reward_ratio=rr,
            margin_requirement=margin,
            max_profit_price=mp_price,
            max_loss_price=ml_price,
        )

    def scenario_analysis(self, legs: List[OptionLeg], S: float,
                          vol_shifts: List[float] = [-0.05, 0, 0.05],
                          spot_shifts: List[float] = [-0.10, -0.05, 0, 0.05, 0.10],
                          ) -> Dict[str, Dict[str, float]]:
        """
        Scenario matrix: P&L under different spot and vol shifts.

        Returns dict[vol_label][spot_label] = P&L
        """
        results = {}
        for dv in vol_shifts:
            vol_label = f"Vol {'+' if dv >= 0 else ''}{dv*100:.0f}%"
            results[vol_label] = {}
            for ds in spot_shifts:
                spot_label = f"Spot {'+' if ds >= 0 else ''}{ds*100:.0f}%"
                new_S = S * (1 + ds)
                # Recalculate with shifted vol
                shifted_legs = []
                for leg in legs:
                    new_leg = OptionLeg(
                        option_type=leg.option_type,
                        strike=leg.strike,
                        expiry_days=leg.expiry_days,
                        side=leg.side,
                        quantity=leg.quantity,
                        premium=leg.premium,
                        iv=max(leg.iv + dv, 0.01),
                    )
                    shifted_legs.append(new_leg)

                # Price each leg at new spot
                total_pnl = 0.0
                for leg, new_leg in zip(legs, shifted_legs):
                    T = new_leg.expiry_days / 365.0
                    new_price = bs_price(new_S, new_leg.strike, T, self.r,
                                        new_leg.iv, new_leg.option_type)
                    entry_cost = leg.premium * leg.side.value * leg.quantity
                    current_val = new_price * leg.side.value * leg.quantity
                    total_pnl += current_val - entry_cost

                results[vol_label][spot_label] = round(total_pnl, 2)

        return results

    def recommend_structure(self, S: float, market_view: str,
                            iv_level: str = "normal",
                            max_risk: float = 5000,
                            dte: int = 30) -> List[Dict]:
        """
        Recommend option structures based on market view and conditions.

        Args:
            S: Current spot price
            market_view: "bullish", "bearish", "neutral", "volatile"
            iv_level: "low", "normal", "high"
            max_risk: Maximum acceptable loss
            dte: Days to expiration
        """
        from strategies.options.structures.families import STRUCTURE_REGISTRY

        recommendations = []
        for sid, sdef in STRUCTURE_REGISTRY.items():
            if sdef.market_view != market_view:
                continue

            # Score based on IV level alignment
            score = 50  # base score
            if iv_level == "high" and sdef.risk_profile == "defined":
                score += 20  # prefer defined risk in high IV
            if iv_level == "low" and "Long" in sdef.name:
                score += 10  # buying cheap options
            if iv_level == "high" and "Short" in sdef.name:
                score += 15  # selling expensive options

            recommendations.append({
                "id": sdef.id,
                "name": sdef.name,
                "family": sdef.family,
                "risk_profile": sdef.risk_profile,
                "score": score,
                "description": sdef.description,
            })

        recommendations.sort(key=lambda x: x["score"], reverse=True)
        return recommendations[:10]
