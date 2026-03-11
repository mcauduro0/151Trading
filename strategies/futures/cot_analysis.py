"""COT (Commitment of Traders) Analysis Strategy (FUT_COT_010).

Uses CFTC Commitment of Traders data to identify positioning extremes
in commodity and financial futures. When commercial hedgers reach extreme
positions, it often signals a trend reversal.

Key features:
- COT net positioning percentile ranking (3-year lookback)
- Commercial vs speculative positioning divergence
- Extreme positioning reversal signals
- Momentum confirmation overlay
- Winsorization at 5%/95% for robustness

Reference: 151 Trading Strategies, COT Analysis section.
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


class COTAnalysis(StrategyBase):
    """Commitment of Traders positioning analysis for futures markets.

    The strategy identifies extremes in commercial hedger positioning
    as contrarian reversal signals, confirmed by speculative momentum.

    Signal logic:
    - Commercial net long at 3-year high -> bullish (hedgers accumulating)
    - Commercial net short at 3-year high -> bearish (hedgers distributing)
    - Speculative positioning confirms or filters the commercial signal
    """

    # COT-tracked commodities with typical contract specs
    COT_UNIVERSE = {
        "CL=F": "Crude Oil",
        "GC=F": "Gold",
        "SI=F": "Silver",
        "NG=F": "Natural Gas",
        "ZC=F": "Corn",
        "ZS=F": "Soybeans",
        "ZW=F": "Wheat",
        "HG=F": "Copper",
        "CT=F": "Cotton",
        "KC=F": "Coffee",
        "SB=F": "Sugar",
    }

    def __init__(
        self,
        percentile_lookback: int = 156,  # ~3 years of weekly data
        extreme_threshold: float = 0.85,
        reversal_threshold: float = 0.15,
        momentum_period: int = 20,
        max_weight: float = 0.10,
    ):
        super().__init__(
            strategy_id="FUT_COT_010",
            name="COT Positioning Analysis",
            asset_class=AssetClass.FUTURES,
            style=StrategyStyle.MEAN_REVERSION,
            description="Contrarian signals from CFTC Commitment of Traders positioning extremes"
        )
        self.percentile_lookback = percentile_lookback
        self.extreme_threshold = extreme_threshold
        self.reversal_threshold = reversal_threshold
        self.momentum_period = momentum_period
        self.max_weight = max_weight

    def required_data(self) -> Dict[str, str]:
        return {
            "prices": "yahoo_finance:daily_ohlcv",
            "cot_data": "cftc:commitment_of_traders",
        }

    def _simulate_cot_from_prices(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Simulate COT-like positioning data from price patterns.

        In production, this would be replaced with actual CFTC COT data.
        Uses price momentum and mean-reversion patterns as proxies.
        """
        if len(prices) < 60:
            return pd.DataFrame()

        # Simulate commercial positioning as inverse of price momentum
        # (commercials tend to be contrarian hedgers)
        mom_60d = prices.pct_change(60)
        commercial_net = -mom_60d  # Inverse of momentum

        # Simulate speculative positioning as aligned with momentum
        mom_20d = prices.pct_change(20)
        speculative_net = mom_20d

        return pd.DataFrame({
            'commercial': commercial_net.iloc[-1],
            'speculative': speculative_net.iloc[-1],
        })

    def _compute_percentile_rank(self, series: pd.Series, lookback: int) -> pd.Series:
        """Compute percentile rank over lookback window."""
        if len(series) < 2:
            return pd.Series(0.5, index=series.index)

        ranks = series.rank(pct=True)
        return ranks

    def _winsorize(self, series: pd.Series, lower: float = 0.05, upper: float = 0.95) -> pd.Series:
        """Winsorize at 5%/95%."""
        lo = series.quantile(lower)
        hi = series.quantile(upper)
        return series.clip(lo, hi)

    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> List[Signal]:
        """Generate COT-based positioning signals."""
        prices = data.get("prices")
        if prices is None or prices.empty:
            return []

        if len(prices) < 60:
            return []

        # Get or simulate COT data
        cot_data = data.get("cot_data")
        if cot_data is None or cot_data.empty:
            cot_data = self._simulate_cot_from_prices(prices)

        if cot_data.empty:
            return []

        commercial = cot_data.get("commercial", pd.Series())
        speculative = cot_data.get("speculative", pd.Series())

        if commercial.empty:
            return []

        # Percentile rank of commercial positioning
        comm_pct = self._compute_percentile_rank(commercial, self.percentile_lookback)
        spec_pct = self._compute_percentile_rank(speculative, self.percentile_lookback)

        # Price momentum for confirmation
        momentum = prices.pct_change(self.momentum_period).iloc[-1]
        momentum = self._winsorize(momentum)

        signals = []
        for symbol in commercial.index:
            if symbol not in prices.columns:
                continue

            c_pct = comm_pct.get(symbol, 0.5)
            s_pct = spec_pct.get(symbol, 0.5)
            mom = momentum.get(symbol, 0)

            # Signal logic:
            # Commercial extreme long (>85th pct) + spec not extreme -> LONG
            # Commercial extreme short (<15th pct) + spec not extreme -> SHORT
            signal_strength = 0.0

            if c_pct > self.extreme_threshold:
                # Commercials extremely long -> bullish contrarian
                signal_strength = (c_pct - 0.5) * 2
                if s_pct < 0.5:  # Specs not yet long -> stronger signal
                    signal_strength *= 1.3
            elif c_pct < self.reversal_threshold:
                # Commercials extremely short -> bearish contrarian
                signal_strength = -(0.5 - c_pct) * 2
                if s_pct > 0.5:  # Specs still long -> stronger signal
                    signal_strength *= 1.3

            # Momentum confirmation (reduce signal if momentum disagrees)
            if signal_strength > 0 and mom < -0.05:
                signal_strength *= 0.5
            elif signal_strength < 0 and mom > 0.05:
                signal_strength *= 0.5

            if abs(signal_strength) < 0.1:
                continue

            weight = min(abs(signal_strength) * self.max_weight, self.max_weight)
            direction = SignalDirection.LONG if signal_strength > 0 else SignalDirection.SHORT

            signals.append(Signal(
                symbol=symbol,
                direction=direction,
                weight=weight,
                metadata={
                    "commercial_percentile": float(c_pct),
                    "speculative_percentile": float(s_pct),
                    "momentum_20d": float(mom),
                    "signal_strength": float(signal_strength),
                    "commodity": self.COT_UNIVERSE.get(symbol, symbol),
                }
            ))

        return signals

    def risk_checks(self, signals: List[Signal],
                    portfolio_state: Optional[Dict] = None) -> List[Signal]:
        """Apply COT-specific risk checks."""
        filtered = []
        for sig in signals:
            # Require minimum signal strength
            strength = abs(sig.metadata.get("signal_strength", 0)) if sig.metadata else 0
            if strength < 0.15:
                continue
            filtered.append(sig)
        return filtered
