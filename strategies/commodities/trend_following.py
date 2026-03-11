"""Commodity Trend Following Strategy (CMD_TF_008).

Implements a multi-timeframe trend following system for commodity futures.
Uses a combination of moving average crossovers, breakout channels, and
momentum indicators to identify and ride commodity trends.

Key features:
- Dual moving average crossover (fast/slow) for trend direction
- Donchian channel breakout for entry confirmation
- ATR-based position sizing for volatility normalization
- Time-series momentum (TSMOM) as the core alpha signal
- Winsorization at 5%/95% for outlier robustness

Reference: 151 Trading Strategies, Commodity Trend Following section.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from strategies.base import (
    StrategyBase, Signal, SignalDirection, AssetClass, StrategyStyle
)


class CommodityTrendFollowing(StrategyBase):
    """Multi-timeframe trend following for commodity futures.

    The strategy combines three signals:
    1. TSMOM (12-month time-series momentum)
    2. MA crossover (50/200 day)
    3. Donchian breakout (20-day channel)

    Position sizing uses inverse-volatility weighting with ATR normalization.
    """

    # Default commodity universe
    DEFAULT_UNIVERSE = [
        "CL=F",   # Crude Oil
        "GC=F",   # Gold
        "SI=F",   # Silver
        "HG=F",   # Copper
        "NG=F",   # Natural Gas
        "ZC=F",   # Corn
        "ZS=F",   # Soybeans
        "ZW=F",   # Wheat
        "CT=F",   # Cotton
        "KC=F",   # Coffee
        "SB=F",   # Sugar
        "PL=F",   # Platinum
    ]

    def __init__(
        self,
        lookback_momentum: int = 252,
        ma_fast: int = 50,
        ma_slow: int = 200,
        donchian_period: int = 20,
        atr_period: int = 20,
        vol_target: float = 0.10,
        max_weight: float = 0.15,
        skip_days: int = 21,
    ):
        super().__init__(
            strategy_id="CMD_TF_008",
            name="Commodity Trend Following",
            asset_class=AssetClass.COMMODITIES,
            style=StrategyStyle.TREND_FOLLOWING,
            description="Multi-timeframe trend following for commodity futures using TSMOM, MA crossover, and Donchian breakout"
        )
        self.lookback_momentum = lookback_momentum
        self.ma_fast = ma_fast
        self.ma_slow = ma_slow
        self.donchian_period = donchian_period
        self.atr_period = atr_period
        self.vol_target = vol_target
        self.max_weight = max_weight
        self.skip_days = skip_days

    def required_data(self) -> Dict[str, str]:
        return {
            "prices": "yahoo_finance:daily_ohlcv",
            "volumes": "yahoo_finance:daily_volume",
        }

    def _compute_tsmom(self, prices: pd.DataFrame) -> pd.Series:
        """Time-series momentum: sign of 12-month return (skip last month)."""
        total_ret = prices.pct_change(self.lookback_momentum).iloc[-1]
        skip_ret = prices.pct_change(self.skip_days).iloc[-1]
        tsmom_ret = total_ret - skip_ret
        return np.sign(tsmom_ret)

    def _compute_ma_signal(self, prices: pd.DataFrame) -> pd.Series:
        """Moving average crossover signal: +1 if fast > slow, -1 otherwise."""
        ma_fast = prices.rolling(self.ma_fast).mean().iloc[-1]
        ma_slow = prices.rolling(self.ma_slow).mean().iloc[-1]
        current = prices.iloc[-1]
        signal = pd.Series(0.0, index=prices.columns)
        signal[current > ma_fast] += 0.5
        signal[current > ma_slow] += 0.5
        signal[current < ma_fast] -= 0.5
        signal[current < ma_slow] -= 0.5
        return signal

    def _compute_donchian_signal(self, prices: pd.DataFrame) -> pd.Series:
        """Donchian channel breakout: +1 if at 20-day high, -1 if at 20-day low."""
        high_channel = prices.rolling(self.donchian_period).max().iloc[-1]
        low_channel = prices.rolling(self.donchian_period).min().iloc[-1]
        current = prices.iloc[-1]
        signal = pd.Series(0.0, index=prices.columns)
        signal[current >= high_channel * 0.98] = 1.0
        signal[current <= low_channel * 1.02] = -1.0
        return signal

    def _compute_atr(self, prices: pd.DataFrame) -> pd.Series:
        """Average True Range for volatility-based position sizing."""
        returns = prices.pct_change().dropna()
        atr = returns.rolling(self.atr_period).std().iloc[-1] * np.sqrt(252)
        return atr

    def _winsorize(self, series: pd.Series, lower: float = 0.05, upper: float = 0.95) -> pd.Series:
        """Winsorize at 5%/95% to reduce outlier impact."""
        lo = series.quantile(lower)
        hi = series.quantile(upper)
        return series.clip(lo, hi)

    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> List[Signal]:
        """Generate trend following signals for commodity universe."""
        prices = data.get("prices")
        if prices is None or prices.empty:
            return []

        if len(prices) < self.lookback_momentum + 10:
            return []

        # Compute three sub-signals
        tsmom = self._compute_tsmom(prices)
        ma_sig = self._compute_ma_signal(prices)
        donchian = self._compute_donchian_signal(prices)

        # Composite signal: weighted average
        composite = 0.4 * tsmom + 0.35 * ma_sig + 0.25 * donchian
        composite = self._winsorize(composite)

        # Volatility-based sizing
        atr = self._compute_atr(prices)
        atr = atr.replace(0, np.nan).fillna(atr.median())

        # Inverse volatility weights
        inv_vol = 1.0 / atr
        inv_vol = inv_vol / inv_vol.sum()

        # Final weights: signal direction * vol-normalized size
        weights = composite * inv_vol
        weights = weights.clip(-self.max_weight, self.max_weight)

        # Normalize to vol target
        portfolio_vol = (weights.abs() * atr).sum()
        if portfolio_vol > 0:
            scale = self.vol_target / portfolio_vol
            weights = weights * scale
            weights = weights.clip(-self.max_weight, self.max_weight)

        # Generate signals
        signals = []
        for symbol, weight in weights.items():
            if abs(weight) < 0.01:
                continue
            direction = SignalDirection.LONG if weight > 0 else SignalDirection.SHORT
            signals.append(Signal(
                symbol=symbol,
                direction=direction,
                weight=abs(weight),
                metadata={
                    "tsmom": float(tsmom.get(symbol, 0)),
                    "ma_signal": float(ma_sig.get(symbol, 0)),
                    "donchian": float(donchian.get(symbol, 0)),
                    "composite": float(composite.get(symbol, 0)),
                    "annualized_vol": float(atr.get(symbol, 0)),
                }
            ))

        return signals

    def risk_checks(self, signals: List[Signal],
                    portfolio_state: Optional[Dict] = None) -> List[Signal]:
        """Apply commodity-specific risk checks."""
        filtered = []
        for sig in signals:
            vol = sig.metadata.get("annualized_vol", 0) if sig.metadata else 0
            # Skip extremely volatile commodities (>80% annualized)
            if vol > 0.80:
                continue
            # Cap weight at max_weight
            if sig.weight > self.max_weight:
                sig.weight = self.max_weight
            filtered.append(sig)
        return filtered
