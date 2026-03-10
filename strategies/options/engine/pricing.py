"""
151 Trading System — Options Pricing Engine
============================================
Core Black-Scholes pricing, Greeks computation, and payoff calculation
for single legs and multi-leg structures.

Supports: European options pricing, full Greeks chain (delta, gamma,
theta, vega, rho), implied volatility solver, and payoff at expiry.
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


class OptionType(Enum):
    CALL = "call"
    PUT = "put"


class PositionSide(Enum):
    LONG = 1
    SHORT = -1


@dataclass
class OptionLeg:
    """Single option leg in a structure."""
    option_type: OptionType
    strike: float
    expiry_days: int          # days to expiration
    side: PositionSide        # long or short
    quantity: int = 1
    premium: float = 0.0      # premium paid/received per contract
    iv: float = 0.0           # implied volatility (annualized)


@dataclass
class Greeks:
    """Option Greeks for a single leg or structure."""
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0        # per day
    vega: float = 0.0         # per 1% vol move
    rho: float = 0.0          # per 1% rate move


@dataclass
class OptionPriceResult:
    """Result of option pricing."""
    price: float
    greeks: Greeks
    iv: float
    intrinsic: float
    time_value: float


# ---------------------------------------------------------------------------
# Black-Scholes Core
# ---------------------------------------------------------------------------

def _d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Calculate d1 in Black-Scholes formula."""
    if T <= 0 or sigma <= 0:
        return 0.0
    return (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))


def _d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Calculate d2 in Black-Scholes formula."""
    if T <= 0 or sigma <= 0:
        return 0.0
    return _d1(S, K, T, r, sigma) - sigma * math.sqrt(T)


def bs_price(S: float, K: float, T: float, r: float, sigma: float,
             option_type: OptionType) -> float:
    """
    Black-Scholes European option price.

    Args:
        S: Spot price
        K: Strike price
        T: Time to expiry (years)
        r: Risk-free rate (annualized)
        sigma: Volatility (annualized)
        option_type: CALL or PUT
    """
    if T <= 0:
        if option_type == OptionType.CALL:
            return max(S - K, 0)
        return max(K - S, 0)

    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(S, K, T, r, sigma)

    if option_type == OptionType.CALL:
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_greeks(S: float, K: float, T: float, r: float, sigma: float,
              option_type: OptionType) -> Greeks:
    """
    Compute full Greeks chain for a European option.
    """
    if T <= 0 or sigma <= 0:
        intrinsic_delta = 1.0 if (option_type == OptionType.CALL and S > K) else \
                         -1.0 if (option_type == OptionType.PUT and S < K) else 0.0
        return Greeks(delta=intrinsic_delta)

    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(S, K, T, r, sigma)
    sqrt_T = math.sqrt(T)
    pdf_d1 = norm.pdf(d1)
    discount = math.exp(-r * T)

    if option_type == OptionType.CALL:
        delta = norm.cdf(d1)
        theta = (-(S * pdf_d1 * sigma) / (2 * sqrt_T)
                 - r * K * discount * norm.cdf(d2)) / 365.0
        rho = K * T * discount * norm.cdf(d2) / 100.0
    else:
        delta = norm.cdf(d1) - 1.0
        theta = (-(S * pdf_d1 * sigma) / (2 * sqrt_T)
                 + r * K * discount * norm.cdf(-d2)) / 365.0
        rho = -K * T * discount * norm.cdf(-d2) / 100.0

    gamma = pdf_d1 / (S * sigma * sqrt_T)
    vega = S * pdf_d1 * sqrt_T / 100.0  # per 1% move

    return Greeks(delta=delta, gamma=gamma, theta=theta, vega=vega, rho=rho)


def implied_volatility(market_price: float, S: float, K: float, T: float,
                       r: float, option_type: OptionType,
                       tol: float = 1e-6) -> float:
    """
    Solve for implied volatility using Brent's method.
    """
    if T <= 0:
        return 0.0

    intrinsic = max(S - K, 0) if option_type == OptionType.CALL else max(K - S, 0)
    if market_price <= intrinsic:
        return 0.001

    def objective(sigma):
        return bs_price(S, K, T, r, sigma, option_type) - market_price

    try:
        iv = brentq(objective, 0.001, 5.0, xtol=tol)
        return iv
    except (ValueError, RuntimeError):
        return 0.0


# ---------------------------------------------------------------------------
# Payoff Engine
# ---------------------------------------------------------------------------

def leg_payoff_at_expiry(leg: OptionLeg, spot_range: np.ndarray) -> np.ndarray:
    """
    Calculate P&L at expiry for a single option leg across spot range.
    Returns net P&L (payoff minus premium paid).
    """
    side = leg.side.value  # +1 or -1
    qty = leg.quantity

    if leg.option_type == OptionType.CALL:
        intrinsic = np.maximum(spot_range - leg.strike, 0)
    else:
        intrinsic = np.maximum(leg.strike - spot_range, 0)

    # Net P&L = side * qty * (intrinsic - premium)
    return side * qty * (intrinsic - leg.premium)


def structure_payoff_at_expiry(legs: List[OptionLeg],
                                spot_range: np.ndarray) -> np.ndarray:
    """
    Calculate combined P&L at expiry for a multi-leg structure.
    """
    total = np.zeros_like(spot_range, dtype=float)
    for leg in legs:
        total += leg_payoff_at_expiry(leg, spot_range)
    return total


def structure_greeks(legs: List[OptionLeg], S: float, r: float = 0.05) -> Greeks:
    """
    Aggregate Greeks for a multi-leg structure.
    """
    total = Greeks()
    for leg in legs:
        T = leg.expiry_days / 365.0
        sigma = leg.iv if leg.iv > 0 else 0.20
        g = bs_greeks(S, leg.strike, T, r, sigma, leg.option_type)
        side = leg.side.value
        qty = leg.quantity
        total.delta += g.delta * side * qty
        total.gamma += g.gamma * side * qty
        total.theta += g.theta * side * qty
        total.vega += g.vega * side * qty
        total.rho += g.rho * side * qty
    return total


def max_profit(legs: List[OptionLeg], spot_range: np.ndarray) -> float:
    """Maximum profit of a structure across spot range."""
    pnl = structure_payoff_at_expiry(legs, spot_range)
    return float(np.max(pnl))


def max_loss(legs: List[OptionLeg], spot_range: np.ndarray) -> float:
    """Maximum loss of a structure across spot range."""
    pnl = structure_payoff_at_expiry(legs, spot_range)
    return float(np.min(pnl))


def breakeven_points(legs: List[OptionLeg], spot_range: np.ndarray) -> List[float]:
    """Find breakeven points where P&L crosses zero."""
    pnl = structure_payoff_at_expiry(legs, spot_range)
    sign_changes = np.where(np.diff(np.sign(pnl)))[0]
    breakevens = []
    for idx in sign_changes:
        # Linear interpolation
        x1, x2 = spot_range[idx], spot_range[idx + 1]
        y1, y2 = pnl[idx], pnl[idx + 1]
        if y2 != y1:
            be = x1 - y1 * (x2 - x1) / (y2 - y1)
            breakevens.append(round(float(be), 2))
    return breakevens


def probability_of_profit(legs: List[OptionLeg], S: float,
                          sigma: float = 0.20, T: float = 30/365,
                          r: float = 0.05, n_sims: int = 10000) -> float:
    """
    Monte Carlo estimate of probability of profit at expiry.
    """
    np.random.seed(42)
    z = np.random.standard_normal(n_sims)
    ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * math.sqrt(T) * z)
    pnl = structure_payoff_at_expiry(legs, ST)
    return float(np.mean(pnl > 0))
