"""Strategy Abstract Base Class.

Every trading strategy in the 151Trading system must implement this interface.
This ensures consistent behavior across backtesting, paper trading, and live execution.

The interface follows the architecture document (D5 Section 5.2):
- generate_features() -> feature frame
- generate_signal() -> raw alpha vector or target signal
- size_positions() -> target holdings
- check_risk() -> pass, fail, clipped targets, or escalated alert
- build_orders() -> order intents
- on_fill() -> updated internal state
- get_metadata() -> tags, assumptions, needed data, expected capacity, known failure modes
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


class StrategyStatus(str, Enum):
    """Strategy lifecycle states."""
    RESEARCH_ONLY = "research_only"
    PAPER_ENABLED = "paper_enabled"
    LIVE_ENABLED = "live_enabled"
    PAUSED = "paused"
    RETIRED = "retired"


class AssetClass(str, Enum):
    """Supported asset classes."""
    EQUITY = "equity"
    ETF = "etf"
    OPTIONS = "options"
    VOLATILITY = "volatility"
    FIXED_INCOME = "fixed_income"
    FX = "fx"
    COMMODITIES = "commodities"
    FUTURES = "futures"
    CRYPTO = "crypto"
    MACRO = "macro"
    STRUCTURED = "structured"
    CONVERTIBLE = "convertible"


class StrategyStyle(str, Enum):
    """Strategy style classification."""
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    VALUE = "value"
    CARRY = "carry"
    VOLATILITY_ARB = "volatility_arb"
    STATISTICAL_ARB = "statistical_arb"
    EVENT_DRIVEN = "event_driven"
    TREND_FOLLOWING = "trend_following"
    FACTOR = "factor"
    ML_BASED = "ml_based"
    RELATIVE_VALUE = "relative_value"
    DIRECTIONAL = "directional"


@dataclass
class StrategyMetadata:
    """Strategy metadata as required by D5."""
    code: str
    name: str
    source_book: str  # "151TS" or "FA"
    asset_class: AssetClass
    style: StrategyStyle
    sub_style: Optional[str] = None
    horizon: str = "daily"  # intraday, daily, weekly, monthly
    directionality: str = "long_short"  # long_only, short_only, long_short, market_neutral
    complexity: str = "moderate"  # simple, moderate, complex, institutional
    description: str = ""
    math_formula: str = ""
    assumptions: List[str] = field(default_factory=list)
    known_failure_modes: List[str] = field(default_factory=list)
    capacity_notes: str = ""
    required_data: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    parameter_bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)


@dataclass
class RiskCheckResult:
    """Result of a pre-trade risk check."""
    passed: bool
    clipped_targets: Optional[pd.Series] = None
    hard_breaches: List[str] = field(default_factory=list)
    soft_warnings: List[str] = field(default_factory=list)
    escalated: bool = False


@dataclass
class OrderIntent:
    """An order intent to be submitted to the OMS."""
    instrument_id: int
    symbol: str
    side: str  # "buy" or "sell"
    qty: float
    order_type: str = "market"
    limit_price: Optional[float] = None
    execution_style: str = "market_on_close"
    urgency: str = "normal"


@dataclass
class StrategyContext:
    """Context passed to strategy methods during execution."""
    as_of_date: date
    universe: List[str]
    parameters: Dict[str, Any]
    current_positions: Optional[pd.Series] = None
    risk_limits: Optional[Dict[str, float]] = None
    market_state: Optional[Dict[str, Any]] = None


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies.

    Every strategy must implement the core pipeline:
    1. generate_features() - Compute features from raw data
    2. generate_signal() - Produce alpha/signal from features
    3. size_positions() - Convert signal to target holdings
    4. check_risk() - Validate targets against risk limits
    5. build_orders() - Create order intents from position deltas
    6. on_fill() - Update state when fills arrive
    7. get_metadata() - Return strategy metadata
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self._params = params or {}
        self._state: Dict[str, Any] = {}

    @abstractmethod
    def get_metadata(self) -> StrategyMetadata:
        """Return strategy metadata including assumptions and failure modes."""
        pass

    @abstractmethod
    def generate_features(self, context: StrategyContext, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Compute features from raw market data.

        Args:
            context: Strategy execution context with date, universe, params
            data: Dictionary of DataFrames keyed by data type (e.g., "bars_1d", "fundamentals")

        Returns:
            DataFrame with computed features, indexed by instrument
        """
        pass

    @abstractmethod
    def generate_signal(self, features: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """Produce raw alpha vector or target signal from features.

        Args:
            features: DataFrame of computed features
            params: Strategy parameters

        Returns:
            Series of raw signal values indexed by instrument
        """
        pass

    def size_positions(
        self,
        signal: pd.Series,
        risk_context: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        """Convert raw signal to target holdings (position sizes).

        Default implementation: normalize signal to sum to book_size.
        Override for custom sizing logic.

        Args:
            signal: Raw alpha/signal values
            risk_context: Current risk state (gross, net, limits)
            params: Strategy parameters

        Returns:
            Series of target position sizes (in dollars or shares)
        """
        book_size = params.get("book_size", 1_000_000)
        max_single_weight = params.get("max_single_weight", 0.05)

        # Normalize to weights
        if signal.abs().sum() > 0:
            weights = signal / signal.abs().sum()
        else:
            return pd.Series(0, index=signal.index)

        # Clip to max single-name weight
        weights = weights.clip(-max_single_weight, max_single_weight)

        # Scale to book size
        targets = weights * book_size
        return targets

    def check_risk(
        self,
        targets: pd.Series,
        risk_context: Dict[str, Any],
    ) -> RiskCheckResult:
        """Validate proposed targets against risk limits.

        Default implementation checks basic limits.
        Override for strategy-specific risk checks.
        """
        result = RiskCheckResult(passed=True)
        limits = risk_context.get("limits", {})

        # Max gross check
        max_gross = limits.get("max_gross", float("inf"))
        gross = targets.abs().sum()
        if gross > max_gross:
            result.soft_warnings.append(f"Gross {gross:.0f} exceeds limit {max_gross:.0f}")
            # Clip proportionally
            scale = max_gross / gross
            result.clipped_targets = targets * scale

        # Max single-name check
        max_name = limits.get("max_single_name", float("inf"))
        violations = targets[targets.abs() > max_name]
        if len(violations) > 0:
            result.soft_warnings.append(f"{len(violations)} names exceed single-name limit")
            clipped = targets.clip(-max_name, max_name)
            result.clipped_targets = clipped

        # Stale data check
        stale_symbols = risk_context.get("stale_symbols", [])
        stale_targets = targets[targets.index.isin(stale_symbols)]
        if len(stale_targets) > 0:
            result.hard_breaches.append(f"Stale data for: {list(stale_targets.index)}")
            result.passed = False

        return result

    def build_orders(
        self,
        current_positions: pd.Series,
        targets: pd.Series,
        market_state: Optional[Dict[str, Any]] = None,
    ) -> List[OrderIntent]:
        """Create order intents from position deltas.

        Args:
            current_positions: Current holdings
            targets: Target holdings after risk checks
            market_state: Current market state (prices, etc.)

        Returns:
            List of OrderIntent objects
        """
        # Compute deltas
        all_symbols = set(current_positions.index) | set(targets.index)
        orders = []

        for symbol in all_symbols:
            current = current_positions.get(symbol, 0)
            target = targets.get(symbol, 0)
            delta = target - current

            if abs(delta) < 1:  # Minimum order threshold
                continue

            orders.append(OrderIntent(
                instrument_id=0,  # Will be resolved by OMS
                symbol=symbol,
                side="buy" if delta > 0 else "sell",
                qty=abs(delta),
            ))

        return orders

    def on_fill(self, fill_event: Dict[str, Any]) -> None:
        """Update internal state when a fill arrives.

        Override if the strategy needs to track fills for state management.
        """
        pass


# ===========================================================================
# Simplified Strategy Interface (for signal-based strategies)
# ===========================================================================

class SignalDirection(str, Enum):
    """Direction of a trading signal."""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


@dataclass
class Signal:
    """A trading signal produced by a strategy."""
    symbol: str
    direction: SignalDirection
    weight: float = 0.0
    metadata: Optional[Dict[str, Any]] = None


class StrategyBase(ABC):
    """Simplified base class for signal-based strategies.

    This is a lighter interface than BaseStrategy, designed for strategies
    that produce discrete signals (long/short/flat) rather than continuous
    alpha vectors. Used by volatility, options, and macro strategies.
    """

    def __init__(self, strategy_id: str, name: str, asset_class: AssetClass,
                 style: StrategyStyle, description: str = ""):
        self.strategy_id = strategy_id
        self.name = name
        self.asset_class = asset_class
        self.style = style
        self.description = description

    @abstractmethod
    def required_data(self) -> Dict[str, str]:
        """Return dict of data requirements: {key: source_spec}."""
        pass

    @abstractmethod
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> List[Signal]:
        """Generate trading signals from input data."""
        pass

    def risk_checks(self, signals: List[Signal],
                    portfolio_state: Optional[Dict] = None) -> List[Signal]:
        """Apply risk management filters to signals. Override for custom logic."""
        return signals


    def validate_params(self, params: Dict[str, Any]) -> List[str]:
        """Validate strategy parameters against bounds.

        Returns list of validation errors.
        """
        metadata = self.get_metadata()
        errors = []

        for param_name, (lower, upper) in metadata.parameter_bounds.items():
            if param_name in params:
                value = params[param_name]
                if value < lower or value > upper:
                    errors.append(
                        f"Parameter '{param_name}' = {value} is outside bounds [{lower}, {upper}]"
                    )

        return errors
