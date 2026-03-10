"""
151 Trading System — Option Structure Families
================================================
58 named option structures organized into 8 families.
Each structure is a factory function returning a list of OptionLeg objects.

Families:
  1. Single Legs (4 structures)
  2. Vertical Spreads (8 structures)
  3. Butterflies & Condors (10 structures)
  4. Straddles & Strangles (8 structures)
  5. Calendar & Diagonal Spreads (8 structures)
  6. Ratio Spreads (8 structures)
  7. Synthetic & Conversion (6 structures)
  8. Complex / Exotic Combos (6 structures)

Total: 58 structures
"""

from dataclasses import dataclass
from typing import List, Dict, Callable, Optional
from strategies.options.engine.pricing import (
    OptionLeg, OptionType, PositionSide
)


@dataclass
class StructureDefinition:
    """Metadata for a named option structure."""
    id: str
    name: str
    family: str
    description: str
    max_legs: int
    builder: Callable  # factory function
    risk_profile: str  # "defined" or "undefined"
    market_view: str   # "bullish", "bearish", "neutral", "volatile"


# Registry of all structures
STRUCTURE_REGISTRY: Dict[str, StructureDefinition] = {}


def register(struct_def: StructureDefinition):
    """Register a structure definition."""
    STRUCTURE_REGISTRY[struct_def.id] = struct_def
    return struct_def.builder


# ===========================================================================
# FAMILY 1: Single Legs (4 structures)
# ===========================================================================

def long_call(S: float, K: float, dte: int, premium: float, iv: float = 0.20) -> List[OptionLeg]:
    return [OptionLeg(OptionType.CALL, K, dte, PositionSide.LONG, 1, premium, iv)]

def short_call(S: float, K: float, dte: int, premium: float, iv: float = 0.20) -> List[OptionLeg]:
    return [OptionLeg(OptionType.CALL, K, dte, PositionSide.SHORT, 1, premium, iv)]

def long_put(S: float, K: float, dte: int, premium: float, iv: float = 0.20) -> List[OptionLeg]:
    return [OptionLeg(OptionType.PUT, K, dte, PositionSide.LONG, 1, premium, iv)]

def short_put(S: float, K: float, dte: int, premium: float, iv: float = 0.20) -> List[OptionLeg]:
    return [OptionLeg(OptionType.PUT, K, dte, PositionSide.SHORT, 1, premium, iv)]

for _id, _name, _fn, _view in [
    ("SL_01", "Long Call", long_call, "bullish"),
    ("SL_02", "Short Call", short_call, "bearish"),
    ("SL_03", "Long Put", long_put, "bearish"),
    ("SL_04", "Short Put", short_put, "bullish"),
]:
    register(StructureDefinition(_id, _name, "Single Legs", f"Single {_name}", 1, _fn, "undefined", _view))


# ===========================================================================
# FAMILY 2: Vertical Spreads (8 structures)
# ===========================================================================

def bull_call_spread(S: float, K_low: float, K_high: float, dte: int,
                     prem_low: float, prem_high: float, iv: float = 0.20) -> List[OptionLeg]:
    return [
        OptionLeg(OptionType.CALL, K_low, dte, PositionSide.LONG, 1, prem_low, iv),
        OptionLeg(OptionType.CALL, K_high, dte, PositionSide.SHORT, 1, prem_high, iv),
    ]

def bear_call_spread(S: float, K_low: float, K_high: float, dte: int,
                     prem_low: float, prem_high: float, iv: float = 0.20) -> List[OptionLeg]:
    return [
        OptionLeg(OptionType.CALL, K_low, dte, PositionSide.SHORT, 1, prem_low, iv),
        OptionLeg(OptionType.CALL, K_high, dte, PositionSide.LONG, 1, prem_high, iv),
    ]

def bull_put_spread(S: float, K_low: float, K_high: float, dte: int,
                    prem_low: float, prem_high: float, iv: float = 0.20) -> List[OptionLeg]:
    return [
        OptionLeg(OptionType.PUT, K_low, dte, PositionSide.SHORT, 1, prem_low, iv),
        OptionLeg(OptionType.PUT, K_high, dte, PositionSide.LONG, 1, prem_high, iv * 1.02),
    ]

def bear_put_spread(S: float, K_low: float, K_high: float, dte: int,
                    prem_low: float, prem_high: float, iv: float = 0.20) -> List[OptionLeg]:
    return [
        OptionLeg(OptionType.PUT, K_low, dte, PositionSide.LONG, 1, prem_low, iv),
        OptionLeg(OptionType.PUT, K_high, dte, PositionSide.SHORT, 1, prem_high, iv),
    ]

def debit_call_spread(S: float, K_low: float, K_high: float, dte: int,
                      prem_low: float, prem_high: float, iv: float = 0.20) -> List[OptionLeg]:
    return bull_call_spread(S, K_low, K_high, dte, prem_low, prem_high, iv)

def credit_call_spread(S: float, K_low: float, K_high: float, dte: int,
                       prem_low: float, prem_high: float, iv: float = 0.20) -> List[OptionLeg]:
    return bear_call_spread(S, K_low, K_high, dte, prem_low, prem_high, iv)

def debit_put_spread(S: float, K_low: float, K_high: float, dte: int,
                     prem_low: float, prem_high: float, iv: float = 0.20) -> List[OptionLeg]:
    return bear_put_spread(S, K_low, K_high, dte, prem_low, prem_high, iv)

def credit_put_spread(S: float, K_low: float, K_high: float, dte: int,
                      prem_low: float, prem_high: float, iv: float = 0.20) -> List[OptionLeg]:
    return bull_put_spread(S, K_low, K_high, dte, prem_low, prem_high, iv)

for _id, _name, _fn, _view in [
    ("VS_01", "Bull Call Spread", bull_call_spread, "bullish"),
    ("VS_02", "Bear Call Spread", bear_call_spread, "bearish"),
    ("VS_03", "Bull Put Spread", bull_put_spread, "bullish"),
    ("VS_04", "Bear Put Spread", bear_put_spread, "bearish"),
    ("VS_05", "Debit Call Spread", debit_call_spread, "bullish"),
    ("VS_06", "Credit Call Spread", credit_call_spread, "bearish"),
    ("VS_07", "Debit Put Spread", debit_put_spread, "bearish"),
    ("VS_08", "Credit Put Spread", credit_put_spread, "bullish"),
]:
    register(StructureDefinition(_id, _name, "Vertical Spreads", f"{_name} vertical", 2, _fn, "defined", _view))


# ===========================================================================
# FAMILY 3: Butterflies & Condors (10 structures)
# ===========================================================================

def long_call_butterfly(S: float, K1: float, K2: float, K3: float, dte: int,
                        p1: float, p2: float, p3: float, iv: float = 0.20) -> List[OptionLeg]:
    return [
        OptionLeg(OptionType.CALL, K1, dte, PositionSide.LONG, 1, p1, iv),
        OptionLeg(OptionType.CALL, K2, dte, PositionSide.SHORT, 2, p2, iv),
        OptionLeg(OptionType.CALL, K3, dte, PositionSide.LONG, 1, p3, iv),
    ]

def short_call_butterfly(S: float, K1: float, K2: float, K3: float, dte: int,
                         p1: float, p2: float, p3: float, iv: float = 0.20) -> List[OptionLeg]:
    return [
        OptionLeg(OptionType.CALL, K1, dte, PositionSide.SHORT, 1, p1, iv),
        OptionLeg(OptionType.CALL, K2, dte, PositionSide.LONG, 2, p2, iv),
        OptionLeg(OptionType.CALL, K3, dte, PositionSide.SHORT, 1, p3, iv),
    ]

def long_put_butterfly(S: float, K1: float, K2: float, K3: float, dte: int,
                       p1: float, p2: float, p3: float, iv: float = 0.20) -> List[OptionLeg]:
    return [
        OptionLeg(OptionType.PUT, K1, dte, PositionSide.LONG, 1, p1, iv),
        OptionLeg(OptionType.PUT, K2, dte, PositionSide.SHORT, 2, p2, iv),
        OptionLeg(OptionType.PUT, K3, dte, PositionSide.LONG, 1, p3, iv),
    ]

def iron_butterfly(S: float, K1: float, K2: float, K3: float, dte: int,
                   p1: float, p2c: float, p2p: float, p3: float, iv: float = 0.20) -> List[OptionLeg]:
    return [
        OptionLeg(OptionType.PUT, K1, dte, PositionSide.LONG, 1, p1, iv),
        OptionLeg(OptionType.PUT, K2, dte, PositionSide.SHORT, 1, p2p, iv),
        OptionLeg(OptionType.CALL, K2, dte, PositionSide.SHORT, 1, p2c, iv),
        OptionLeg(OptionType.CALL, K3, dte, PositionSide.LONG, 1, p3, iv),
    ]

def long_call_condor(S: float, K1: float, K2: float, K3: float, K4: float, dte: int,
                     p1: float, p2: float, p3: float, p4: float, iv: float = 0.20) -> List[OptionLeg]:
    return [
        OptionLeg(OptionType.CALL, K1, dte, PositionSide.LONG, 1, p1, iv),
        OptionLeg(OptionType.CALL, K2, dte, PositionSide.SHORT, 1, p2, iv),
        OptionLeg(OptionType.CALL, K3, dte, PositionSide.SHORT, 1, p3, iv),
        OptionLeg(OptionType.CALL, K4, dte, PositionSide.LONG, 1, p4, iv),
    ]

def short_call_condor(S: float, K1: float, K2: float, K3: float, K4: float, dte: int,
                      p1: float, p2: float, p3: float, p4: float, iv: float = 0.20) -> List[OptionLeg]:
    return [
        OptionLeg(OptionType.CALL, K1, dte, PositionSide.SHORT, 1, p1, iv),
        OptionLeg(OptionType.CALL, K2, dte, PositionSide.LONG, 1, p2, iv),
        OptionLeg(OptionType.CALL, K3, dte, PositionSide.LONG, 1, p3, iv),
        OptionLeg(OptionType.CALL, K4, dte, PositionSide.SHORT, 1, p4, iv),
    ]

def iron_condor(S: float, K1: float, K2: float, K3: float, K4: float, dte: int,
                p1: float, p2: float, p3: float, p4: float, iv: float = 0.20) -> List[OptionLeg]:
    return [
        OptionLeg(OptionType.PUT, K1, dte, PositionSide.LONG, 1, p1, iv),
        OptionLeg(OptionType.PUT, K2, dte, PositionSide.SHORT, 1, p2, iv),
        OptionLeg(OptionType.CALL, K3, dte, PositionSide.SHORT, 1, p3, iv),
        OptionLeg(OptionType.CALL, K4, dte, PositionSide.LONG, 1, p4, iv),
    ]

def reverse_iron_condor(S: float, K1: float, K2: float, K3: float, K4: float, dte: int,
                        p1: float, p2: float, p3: float, p4: float, iv: float = 0.20) -> List[OptionLeg]:
    return [
        OptionLeg(OptionType.PUT, K1, dte, PositionSide.SHORT, 1, p1, iv),
        OptionLeg(OptionType.PUT, K2, dte, PositionSide.LONG, 1, p2, iv),
        OptionLeg(OptionType.CALL, K3, dte, PositionSide.LONG, 1, p3, iv),
        OptionLeg(OptionType.CALL, K4, dte, PositionSide.SHORT, 1, p4, iv),
    ]

def broken_wing_butterfly_call(S: float, K1: float, K2: float, K3: float, dte: int,
                               p1: float, p2: float, p3: float, iv: float = 0.20) -> List[OptionLeg]:
    """Broken wing: K3 is further OTM than symmetric."""
    return [
        OptionLeg(OptionType.CALL, K1, dte, PositionSide.LONG, 1, p1, iv),
        OptionLeg(OptionType.CALL, K2, dte, PositionSide.SHORT, 2, p2, iv),
        OptionLeg(OptionType.CALL, K3, dte, PositionSide.LONG, 1, p3, iv),
    ]

def broken_wing_butterfly_put(S: float, K1: float, K2: float, K3: float, dte: int,
                              p1: float, p2: float, p3: float, iv: float = 0.20) -> List[OptionLeg]:
    return [
        OptionLeg(OptionType.PUT, K1, dte, PositionSide.LONG, 1, p1, iv),
        OptionLeg(OptionType.PUT, K2, dte, PositionSide.SHORT, 2, p2, iv),
        OptionLeg(OptionType.PUT, K3, dte, PositionSide.LONG, 1, p3, iv),
    ]

for _id, _name, _fn, _view, _legs in [
    ("BC_01", "Long Call Butterfly", long_call_butterfly, "neutral", 3),
    ("BC_02", "Short Call Butterfly", short_call_butterfly, "volatile", 3),
    ("BC_03", "Long Put Butterfly", long_put_butterfly, "neutral", 3),
    ("BC_04", "Iron Butterfly", iron_butterfly, "neutral", 4),
    ("BC_05", "Long Call Condor", long_call_condor, "neutral", 4),
    ("BC_06", "Short Call Condor", short_call_condor, "volatile", 4),
    ("BC_07", "Iron Condor", iron_condor, "neutral", 4),
    ("BC_08", "Reverse Iron Condor", reverse_iron_condor, "volatile", 4),
    ("BC_09", "Broken Wing Butterfly (Call)", broken_wing_butterfly_call, "neutral", 3),
    ("BC_10", "Broken Wing Butterfly (Put)", broken_wing_butterfly_put, "neutral", 3),
]:
    register(StructureDefinition(_id, _name, "Butterflies & Condors", f"{_name}", _legs, _fn, "defined", _view))


# ===========================================================================
# FAMILY 4: Straddles & Strangles (8 structures)
# ===========================================================================

def long_straddle(S: float, K: float, dte: int, pc: float, pp: float, iv: float = 0.20) -> List[OptionLeg]:
    return [
        OptionLeg(OptionType.CALL, K, dte, PositionSide.LONG, 1, pc, iv),
        OptionLeg(OptionType.PUT, K, dte, PositionSide.LONG, 1, pp, iv),
    ]

def short_straddle(S: float, K: float, dte: int, pc: float, pp: float, iv: float = 0.20) -> List[OptionLeg]:
    return [
        OptionLeg(OptionType.CALL, K, dte, PositionSide.SHORT, 1, pc, iv),
        OptionLeg(OptionType.PUT, K, dte, PositionSide.SHORT, 1, pp, iv),
    ]

def long_strangle(S: float, K_put: float, K_call: float, dte: int,
                  pc: float, pp: float, iv: float = 0.20) -> List[OptionLeg]:
    return [
        OptionLeg(OptionType.PUT, K_put, dte, PositionSide.LONG, 1, pp, iv),
        OptionLeg(OptionType.CALL, K_call, dte, PositionSide.LONG, 1, pc, iv),
    ]

def short_strangle(S: float, K_put: float, K_call: float, dte: int,
                   pc: float, pp: float, iv: float = 0.20) -> List[OptionLeg]:
    return [
        OptionLeg(OptionType.PUT, K_put, dte, PositionSide.SHORT, 1, pp, iv),
        OptionLeg(OptionType.CALL, K_call, dte, PositionSide.SHORT, 1, pc, iv),
    ]

def strap(S: float, K: float, dte: int, pc: float, pp: float, iv: float = 0.20) -> List[OptionLeg]:
    """Strap: 2 calls + 1 put at same strike (bullish bias)."""
    return [
        OptionLeg(OptionType.CALL, K, dte, PositionSide.LONG, 2, pc, iv),
        OptionLeg(OptionType.PUT, K, dte, PositionSide.LONG, 1, pp, iv),
    ]

def strip(S: float, K: float, dte: int, pc: float, pp: float, iv: float = 0.20) -> List[OptionLeg]:
    """Strip: 1 call + 2 puts at same strike (bearish bias)."""
    return [
        OptionLeg(OptionType.CALL, K, dte, PositionSide.LONG, 1, pc, iv),
        OptionLeg(OptionType.PUT, K, dte, PositionSide.LONG, 2, pp, iv),
    ]

def guts(S: float, K_low: float, K_high: float, dte: int,
         pc: float, pp: float, iv: float = 0.20) -> List[OptionLeg]:
    """Long Guts: ITM call + ITM put."""
    return [
        OptionLeg(OptionType.CALL, K_low, dte, PositionSide.LONG, 1, pc, iv),
        OptionLeg(OptionType.PUT, K_high, dte, PositionSide.LONG, 1, pp, iv),
    ]

def short_guts(S: float, K_low: float, K_high: float, dte: int,
               pc: float, pp: float, iv: float = 0.20) -> List[OptionLeg]:
    return [
        OptionLeg(OptionType.CALL, K_low, dte, PositionSide.SHORT, 1, pc, iv),
        OptionLeg(OptionType.PUT, K_high, dte, PositionSide.SHORT, 1, pp, iv),
    ]

for _id, _name, _fn, _view in [
    ("SS_01", "Long Straddle", long_straddle, "volatile"),
    ("SS_02", "Short Straddle", short_straddle, "neutral"),
    ("SS_03", "Long Strangle", long_strangle, "volatile"),
    ("SS_04", "Short Strangle", short_strangle, "neutral"),
    ("SS_05", "Strap", strap, "bullish"),
    ("SS_06", "Strip", strip, "bearish"),
    ("SS_07", "Long Guts", guts, "volatile"),
    ("SS_08", "Short Guts", short_guts, "neutral"),
]:
    register(StructureDefinition(_id, _name, "Straddles & Strangles", f"{_name}", 2, _fn,
                                  "undefined" if "Long" in _name or _name in ("Strap", "Strip") else "undefined", _view))


# ===========================================================================
# FAMILY 5: Calendar & Diagonal Spreads (8 structures)
# ===========================================================================

def long_call_calendar(S: float, K: float, dte_near: int, dte_far: int,
                       p_near: float, p_far: float, iv: float = 0.20) -> List[OptionLeg]:
    return [
        OptionLeg(OptionType.CALL, K, dte_near, PositionSide.SHORT, 1, p_near, iv),
        OptionLeg(OptionType.CALL, K, dte_far, PositionSide.LONG, 1, p_far, iv * 0.95),
    ]

def short_call_calendar(S: float, K: float, dte_near: int, dte_far: int,
                        p_near: float, p_far: float, iv: float = 0.20) -> List[OptionLeg]:
    return [
        OptionLeg(OptionType.CALL, K, dte_near, PositionSide.LONG, 1, p_near, iv),
        OptionLeg(OptionType.CALL, K, dte_far, PositionSide.SHORT, 1, p_far, iv * 0.95),
    ]

def long_put_calendar(S: float, K: float, dte_near: int, dte_far: int,
                      p_near: float, p_far: float, iv: float = 0.20) -> List[OptionLeg]:
    return [
        OptionLeg(OptionType.PUT, K, dte_near, PositionSide.SHORT, 1, p_near, iv),
        OptionLeg(OptionType.PUT, K, dte_far, PositionSide.LONG, 1, p_far, iv * 0.95),
    ]

def short_put_calendar(S: float, K: float, dte_near: int, dte_far: int,
                       p_near: float, p_far: float, iv: float = 0.20) -> List[OptionLeg]:
    return [
        OptionLeg(OptionType.PUT, K, dte_near, PositionSide.LONG, 1, p_near, iv),
        OptionLeg(OptionType.PUT, K, dte_far, PositionSide.SHORT, 1, p_far, iv * 0.95),
    ]

def call_diagonal_bull(S: float, K_near: float, K_far: float, dte_near: int, dte_far: int,
                       p_near: float, p_far: float, iv: float = 0.20) -> List[OptionLeg]:
    return [
        OptionLeg(OptionType.CALL, K_near, dte_near, PositionSide.SHORT, 1, p_near, iv),
        OptionLeg(OptionType.CALL, K_far, dte_far, PositionSide.LONG, 1, p_far, iv * 0.95),
    ]

def call_diagonal_bear(S: float, K_near: float, K_far: float, dte_near: int, dte_far: int,
                       p_near: float, p_far: float, iv: float = 0.20) -> List[OptionLeg]:
    return [
        OptionLeg(OptionType.CALL, K_near, dte_near, PositionSide.LONG, 1, p_near, iv),
        OptionLeg(OptionType.CALL, K_far, dte_far, PositionSide.SHORT, 1, p_far, iv * 0.95),
    ]

def put_diagonal_bull(S: float, K_near: float, K_far: float, dte_near: int, dte_far: int,
                      p_near: float, p_far: float, iv: float = 0.20) -> List[OptionLeg]:
    return [
        OptionLeg(OptionType.PUT, K_near, dte_near, PositionSide.SHORT, 1, p_near, iv),
        OptionLeg(OptionType.PUT, K_far, dte_far, PositionSide.LONG, 1, p_far, iv * 0.95),
    ]

def put_diagonal_bear(S: float, K_near: float, K_far: float, dte_near: int, dte_far: int,
                      p_near: float, p_far: float, iv: float = 0.20) -> List[OptionLeg]:
    return [
        OptionLeg(OptionType.PUT, K_near, dte_near, PositionSide.LONG, 1, p_near, iv),
        OptionLeg(OptionType.PUT, K_far, dte_far, PositionSide.SHORT, 1, p_far, iv * 0.95),
    ]

for _id, _name, _fn, _view in [
    ("CD_01", "Long Call Calendar", long_call_calendar, "neutral"),
    ("CD_02", "Short Call Calendar", short_call_calendar, "volatile"),
    ("CD_03", "Long Put Calendar", long_put_calendar, "neutral"),
    ("CD_04", "Short Put Calendar", short_put_calendar, "volatile"),
    ("CD_05", "Call Diagonal (Bull)", call_diagonal_bull, "bullish"),
    ("CD_06", "Call Diagonal (Bear)", call_diagonal_bear, "bearish"),
    ("CD_07", "Put Diagonal (Bull)", put_diagonal_bull, "bullish"),
    ("CD_08", "Put Diagonal (Bear)", put_diagonal_bear, "bearish"),
]:
    register(StructureDefinition(_id, _name, "Calendar & Diagonal Spreads", f"{_name}", 2, _fn, "defined", _view))


# ===========================================================================
# FAMILY 6: Ratio Spreads (8 structures)
# ===========================================================================

def call_ratio_spread(S: float, K1: float, K2: float, dte: int,
                      p1: float, p2: float, iv: float = 0.20) -> List[OptionLeg]:
    """Buy 1 lower call, sell 2 higher calls."""
    return [
        OptionLeg(OptionType.CALL, K1, dte, PositionSide.LONG, 1, p1, iv),
        OptionLeg(OptionType.CALL, K2, dte, PositionSide.SHORT, 2, p2, iv),
    ]

def put_ratio_spread(S: float, K1: float, K2: float, dte: int,
                     p1: float, p2: float, iv: float = 0.20) -> List[OptionLeg]:
    """Buy 1 higher put, sell 2 lower puts."""
    return [
        OptionLeg(OptionType.PUT, K2, dte, PositionSide.LONG, 1, p2, iv),
        OptionLeg(OptionType.PUT, K1, dte, PositionSide.SHORT, 2, p1, iv),
    ]

def call_ratio_backspread(S: float, K1: float, K2: float, dte: int,
                          p1: float, p2: float, iv: float = 0.20) -> List[OptionLeg]:
    """Sell 1 lower call, buy 2 higher calls."""
    return [
        OptionLeg(OptionType.CALL, K1, dte, PositionSide.SHORT, 1, p1, iv),
        OptionLeg(OptionType.CALL, K2, dte, PositionSide.LONG, 2, p2, iv),
    ]

def put_ratio_backspread(S: float, K1: float, K2: float, dte: int,
                         p1: float, p2: float, iv: float = 0.20) -> List[OptionLeg]:
    """Sell 1 higher put, buy 2 lower puts."""
    return [
        OptionLeg(OptionType.PUT, K2, dte, PositionSide.SHORT, 1, p2, iv),
        OptionLeg(OptionType.PUT, K1, dte, PositionSide.LONG, 2, p1, iv),
    ]

def christmas_tree_call(S: float, K1: float, K2: float, K3: float, dte: int,
                        p1: float, p2: float, p3: float, iv: float = 0.20) -> List[OptionLeg]:
    """Buy 1 ATM call, sell 1 OTM call, sell 1 further OTM call."""
    return [
        OptionLeg(OptionType.CALL, K1, dte, PositionSide.LONG, 1, p1, iv),
        OptionLeg(OptionType.CALL, K2, dte, PositionSide.SHORT, 1, p2, iv),
        OptionLeg(OptionType.CALL, K3, dte, PositionSide.SHORT, 1, p3, iv),
    ]

def christmas_tree_put(S: float, K1: float, K2: float, K3: float, dte: int,
                       p1: float, p2: float, p3: float, iv: float = 0.20) -> List[OptionLeg]:
    return [
        OptionLeg(OptionType.PUT, K3, dte, PositionSide.LONG, 1, p3, iv),
        OptionLeg(OptionType.PUT, K2, dte, PositionSide.SHORT, 1, p2, iv),
        OptionLeg(OptionType.PUT, K1, dte, PositionSide.SHORT, 1, p1, iv),
    ]

def jade_lizard(S: float, K_put: float, K_call_low: float, K_call_high: float, dte: int,
                pp: float, pc_low: float, pc_high: float, iv: float = 0.20) -> List[OptionLeg]:
    """Short put + short call spread (no upside risk)."""
    return [
        OptionLeg(OptionType.PUT, K_put, dte, PositionSide.SHORT, 1, pp, iv),
        OptionLeg(OptionType.CALL, K_call_low, dte, PositionSide.SHORT, 1, pc_low, iv),
        OptionLeg(OptionType.CALL, K_call_high, dte, PositionSide.LONG, 1, pc_high, iv),
    ]

def twisted_sister(S: float, K_put_low: float, K_put_high: float, K_call: float, dte: int,
                   pp_low: float, pp_high: float, pc: float, iv: float = 0.20) -> List[OptionLeg]:
    """Short call + short put spread."""
    return [
        OptionLeg(OptionType.PUT, K_put_low, dte, PositionSide.LONG, 1, pp_low, iv),
        OptionLeg(OptionType.PUT, K_put_high, dte, PositionSide.SHORT, 1, pp_high, iv),
        OptionLeg(OptionType.CALL, K_call, dte, PositionSide.SHORT, 1, pc, iv),
    ]

for _id, _name, _fn, _view in [
    ("RS_01", "Call Ratio Spread", call_ratio_spread, "neutral"),
    ("RS_02", "Put Ratio Spread", put_ratio_spread, "neutral"),
    ("RS_03", "Call Ratio Backspread", call_ratio_backspread, "bullish"),
    ("RS_04", "Put Ratio Backspread", put_ratio_backspread, "bearish"),
    ("RS_05", "Christmas Tree (Call)", christmas_tree_call, "neutral"),
    ("RS_06", "Christmas Tree (Put)", christmas_tree_put, "neutral"),
    ("RS_07", "Jade Lizard", jade_lizard, "neutral"),
    ("RS_08", "Twisted Sister", twisted_sister, "neutral"),
]:
    register(StructureDefinition(_id, _name, "Ratio Spreads", f"{_name}", 3, _fn, "undefined", _view))


# ===========================================================================
# FAMILY 7: Synthetic & Conversion (6 structures)
# ===========================================================================

def synthetic_long(S: float, K: float, dte: int, pc: float, pp: float, iv: float = 0.20) -> List[OptionLeg]:
    """Synthetic long stock: long call + short put at same strike."""
    return [
        OptionLeg(OptionType.CALL, K, dte, PositionSide.LONG, 1, pc, iv),
        OptionLeg(OptionType.PUT, K, dte, PositionSide.SHORT, 1, pp, iv),
    ]

def synthetic_short(S: float, K: float, dte: int, pc: float, pp: float, iv: float = 0.20) -> List[OptionLeg]:
    return [
        OptionLeg(OptionType.CALL, K, dte, PositionSide.SHORT, 1, pc, iv),
        OptionLeg(OptionType.PUT, K, dte, PositionSide.LONG, 1, pp, iv),
    ]

def collar(S: float, K_put: float, K_call: float, dte: int,
           pp: float, pc: float, iv: float = 0.20) -> List[OptionLeg]:
    """Protective collar: long put + short call (with underlying)."""
    return [
        OptionLeg(OptionType.PUT, K_put, dte, PositionSide.LONG, 1, pp, iv),
        OptionLeg(OptionType.CALL, K_call, dte, PositionSide.SHORT, 1, pc, iv),
    ]

def risk_reversal(S: float, K_put: float, K_call: float, dte: int,
                  pp: float, pc: float, iv: float = 0.20) -> List[OptionLeg]:
    """Risk reversal: short OTM put + long OTM call."""
    return [
        OptionLeg(OptionType.PUT, K_put, dte, PositionSide.SHORT, 1, pp, iv),
        OptionLeg(OptionType.CALL, K_call, dte, PositionSide.LONG, 1, pc, iv),
    ]

def conversion(S: float, K: float, dte: int, pc: float, pp: float, iv: float = 0.20) -> List[OptionLeg]:
    """Conversion: long put + short call (with long stock)."""
    return [
        OptionLeg(OptionType.PUT, K, dte, PositionSide.LONG, 1, pp, iv),
        OptionLeg(OptionType.CALL, K, dte, PositionSide.SHORT, 1, pc, iv),
    ]

def reversal(S: float, K: float, dte: int, pc: float, pp: float, iv: float = 0.20) -> List[OptionLeg]:
    """Reversal: short put + long call (with short stock)."""
    return [
        OptionLeg(OptionType.PUT, K, dte, PositionSide.SHORT, 1, pp, iv),
        OptionLeg(OptionType.CALL, K, dte, PositionSide.LONG, 1, pc, iv),
    ]

for _id, _name, _fn, _view in [
    ("SC_01", "Synthetic Long", synthetic_long, "bullish"),
    ("SC_02", "Synthetic Short", synthetic_short, "bearish"),
    ("SC_03", "Collar", collar, "neutral"),
    ("SC_04", "Risk Reversal", risk_reversal, "bullish"),
    ("SC_05", "Conversion", conversion, "neutral"),
    ("SC_06", "Reversal", reversal, "bullish"),
]:
    register(StructureDefinition(_id, _name, "Synthetic & Conversion", f"{_name}", 2, _fn, "defined", _view))


# ===========================================================================
# FAMILY 8: Complex / Exotic Combos (6 structures)
# ===========================================================================

def double_diagonal(S: float, K_put_near: float, K_call_near: float,
                    K_put_far: float, K_call_far: float,
                    dte_near: int, dte_far: int,
                    pp_near: float, pc_near: float,
                    pp_far: float, pc_far: float, iv: float = 0.20) -> List[OptionLeg]:
    """Double diagonal: sell near-term strangle, buy far-term strangle."""
    return [
        OptionLeg(OptionType.PUT, K_put_near, dte_near, PositionSide.SHORT, 1, pp_near, iv),
        OptionLeg(OptionType.CALL, K_call_near, dte_near, PositionSide.SHORT, 1, pc_near, iv),
        OptionLeg(OptionType.PUT, K_put_far, dte_far, PositionSide.LONG, 1, pp_far, iv * 0.95),
        OptionLeg(OptionType.CALL, K_call_far, dte_far, PositionSide.LONG, 1, pc_far, iv * 0.95),
    ]

def double_calendar(S: float, K_put: float, K_call: float,
                    dte_near: int, dte_far: int,
                    pp_near: float, pc_near: float,
                    pp_far: float, pc_far: float, iv: float = 0.20) -> List[OptionLeg]:
    """Double calendar: two calendar spreads (put + call)."""
    return [
        OptionLeg(OptionType.PUT, K_put, dte_near, PositionSide.SHORT, 1, pp_near, iv),
        OptionLeg(OptionType.PUT, K_put, dte_far, PositionSide.LONG, 1, pp_far, iv * 0.95),
        OptionLeg(OptionType.CALL, K_call, dte_near, PositionSide.SHORT, 1, pc_near, iv),
        OptionLeg(OptionType.CALL, K_call, dte_far, PositionSide.LONG, 1, pc_far, iv * 0.95),
    ]

def seagull_bullish(S: float, K1: float, K2: float, K3: float, dte: int,
                    p1: float, p2: float, p3: float, iv: float = 0.20) -> List[OptionLeg]:
    """Bullish seagull: long call spread + short put."""
    return [
        OptionLeg(OptionType.PUT, K1, dte, PositionSide.SHORT, 1, p1, iv),
        OptionLeg(OptionType.CALL, K2, dte, PositionSide.LONG, 1, p2, iv),
        OptionLeg(OptionType.CALL, K3, dte, PositionSide.SHORT, 1, p3, iv),
    ]

def seagull_bearish(S: float, K1: float, K2: float, K3: float, dte: int,
                    p1: float, p2: float, p3: float, iv: float = 0.20) -> List[OptionLeg]:
    """Bearish seagull: long put spread + short call."""
    return [
        OptionLeg(OptionType.PUT, K1, dte, PositionSide.LONG, 1, p1, iv),
        OptionLeg(OptionType.PUT, K2, dte, PositionSide.SHORT, 1, p2, iv),
        OptionLeg(OptionType.CALL, K3, dte, PositionSide.SHORT, 1, p3, iv),
    ]

def zebra(S: float, K1: float, K2: float, dte: int,
          p1: float, p2: float, iv: float = 0.20) -> List[OptionLeg]:
    """ZEBRA (Zero Extrinsic Back Ratio): 2 ATM calls - 1 ITM call."""
    return [
        OptionLeg(OptionType.CALL, K1, dte, PositionSide.SHORT, 1, p1, iv),
        OptionLeg(OptionType.CALL, K2, dte, PositionSide.LONG, 2, p2, iv),
    ]

def big_lizard(S: float, K_put: float, K_call: float, dte: int,
               pp: float, pc: float, iv: float = 0.20) -> List[OptionLeg]:
    """Big Lizard: short straddle + long OTM call for upside protection."""
    return [
        OptionLeg(OptionType.PUT, K_put, dte, PositionSide.SHORT, 1, pp, iv),
        OptionLeg(OptionType.CALL, K_put, dte, PositionSide.SHORT, 1, pc, iv),
        OptionLeg(OptionType.CALL, K_call, dte, PositionSide.LONG, 1, pc * 0.3, iv),
    ]

for _id, _name, _fn, _view in [
    ("CX_01", "Double Diagonal", double_diagonal, "neutral"),
    ("CX_02", "Double Calendar", double_calendar, "neutral"),
    ("CX_03", "Seagull (Bullish)", seagull_bullish, "bullish"),
    ("CX_04", "Seagull (Bearish)", seagull_bearish, "bearish"),
    ("CX_05", "ZEBRA", zebra, "bullish"),
    ("CX_06", "Big Lizard", big_lizard, "neutral"),
]:
    register(StructureDefinition(_id, _name, "Complex / Exotic Combos", f"{_name}", 4, _fn, "defined", _view))


# ===========================================================================
# Public API
# ===========================================================================

def get_all_families() -> Dict[str, List[StructureDefinition]]:
    """Get all structures grouped by family."""
    families: Dict[str, List[StructureDefinition]] = {}
    for sd in STRUCTURE_REGISTRY.values():
        families.setdefault(sd.family, []).append(sd)
    return families


def get_structure(structure_id: str) -> Optional[StructureDefinition]:
    """Get a structure by ID."""
    return STRUCTURE_REGISTRY.get(structure_id)


def list_structures() -> List[StructureDefinition]:
    """List all registered structures."""
    return list(STRUCTURE_REGISTRY.values())


def count_structures() -> int:
    """Count total registered structures."""
    return len(STRUCTURE_REGISTRY)
