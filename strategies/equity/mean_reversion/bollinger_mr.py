"""
EQ_MR_005 — Bollinger Band Mean Reversion Strategy

Exploits short-term mean reversion by trading stocks that deviate
significantly from their moving average, as measured by Bollinger Bands.

Signal Construction:
    1. Compute 20-day SMA and 2σ Bollinger Bands for each stock
    2. Calculate %B = (Price - Lower Band) / (Upper Band - Lower Band)
    3. Compute Bollinger Band Width (BBW) = (Upper - Lower) / SMA
    4. Combine with RSI(14) for confirmation:
       - Long when %B < 0.05 AND RSI < 30 (oversold)
       - Short when %B > 0.95 AND RSI > 70 (overbought)
    5. Signal strength = distance from mean in z-score terms
    6. Apply volume confirmation: require above-average volume on signal day

Position Sizing:
    - Signal-weighted proportional to deviation magnitude
    - Max single-name weight: 3% (tighter for mean reversion)
    - Holding period: 5-10 days (short-term)

Risk Controls:
    - Stop-loss: 3% per position
    - Max concurrent positions: 20
    - Regime filter: skip signals when VIX > 35 (mean reversion fails in crisis)
    - ADV filter: min 20-day avg volume > $5M

Academic basis:
    - Jegadeesh (1990): Short-term return reversal
    - Lehmann (1990): Fama-French short-term contrarian profits
    - Lo & MacKinlay (1990): Mean reversion in stock prices
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional
from datetime import date

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from base import (
    BaseStrategy, StrategyMetadata, StrategyContext,
    AssetClass, StrategyStyle, RiskCheckResult
)


class BollingerMeanReversion(BaseStrategy):
    """
    Bollinger Band Mean Reversion.

    Trades short-term oversold/overbought conditions using Bollinger Bands
    combined with RSI confirmation and volume filters.
    """

    DEFAULT_PARAMS = {
        # Bollinger Band parameters
        "bb_window": 20,             # SMA lookback
        "bb_std": 2.0,               # Standard deviations for bands
        "pct_b_long_threshold": 0.05,  # %B below this → long signal
        "pct_b_short_threshold": 0.95, # %B above this → short signal

        # RSI parameters
        "rsi_period": 14,
        "rsi_oversold": 30,
        "rsi_overbought": 70,
        "use_rsi_confirmation": True,

        # Volume confirmation
        "volume_confirmation": True,
        "volume_ratio_threshold": 1.2,  # Current vol / 20d avg vol

        # Position sizing
        "book_size": 1_000_000,
        "max_single_weight": 0.03,   # Tighter for MR
        "max_positions": 20,
        "holding_period_days": 7,

        # Risk controls
        "stop_loss_pct": 0.03,       # 3% stop-loss
        "max_vix": 35,               # Skip signals in crisis
        "min_price": 10.0,
        "min_adv": 5_000_000,        # Min 20d avg dollar volume

        # Filters
        "min_bbw": 0.02,             # Min bandwidth (avoid low-vol stocks)
        "max_bbw": 0.30,             # Max bandwidth (avoid too volatile)
        "rebalance_frequency": "daily",
    }

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        merged = {**self.DEFAULT_PARAMS, **(params or {})}
        super().__init__(merged)

    def get_metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            code="EQ_MR_005",
            name="Bollinger Band Mean Reversion",
            source_book="151TS",
            asset_class=AssetClass.EQUITY,
            style=StrategyStyle.MEAN_REVERSION,
            sub_style="short-term Bollinger Band reversal",
            horizon="daily",
            directionality="long_short",
            complexity="moderate",
            description=(
                "Trades short-term mean reversion using Bollinger Bands with RSI "
                "confirmation and volume filters. Longs oversold stocks (%B<0.05, "
                "RSI<30), shorts overbought (%B>0.95, RSI>70). 5-10 day holding period."
            ),
            math_formula=(
                "SMA_20 = mean(P, 20); σ = std(P, 20); "
                "Upper = SMA + 2σ; Lower = SMA - 2σ; "
                "%B = (P - Lower)/(Upper - Lower); "
                "RSI = 100 - 100/(1 + avg_gain/avg_loss); "
                "Long: %B < 0.05 AND RSI < 30; Short: %B > 0.95 AND RSI > 70"
            ),
            assumptions=[
                "Short-term price dislocations revert to the mean within 5-10 days",
                "Bollinger Band extremes capture statistically significant deviations",
                "RSI provides additional confirmation of oversold/overbought conditions",
                "Volume spikes at extremes indicate capitulation/exhaustion",
                "Mean reversion works in normal volatility regimes, not during crises",
            ],
            known_failure_modes=[
                "Trending markets: stocks can stay overbought/oversold for extended periods",
                "Regime change: mean reversion fails during momentum crashes or VIX spikes",
                "Earnings/events: fundamental news can override technical mean reversion",
                "Low-vol stocks: narrow bands produce false signals",
                "Gap risk: overnight gaps can blow through stop-losses",
            ],
            capacity_notes="Moderate capacity; short holding period requires frequent trading",
            required_data=["bars_1d", "volume"],
            parameters=self._params,
            parameter_bounds={
                "bb_window": (10, 50),
                "bb_std": (1.0, 3.0),
                "pct_b_long_threshold": (0.0, 0.20),
                "pct_b_short_threshold": (0.80, 1.0),
                "rsi_period": (7, 28),
                "rsi_oversold": (15, 40),
                "rsi_overbought": (60, 85),
                "max_positions": (5, 50),
                "stop_loss_pct": (0.01, 0.10),
                "max_vix": (20, 50),
            },
        )

    def generate_features(self, context: StrategyContext, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Compute Bollinger Band, RSI, and volume features.

        Features generated:
            - sma_20: 20-day simple moving average
            - bb_upper: Upper Bollinger Band
            - bb_lower: Lower Bollinger Band
            - pct_b: %B indicator (0-1 range, can exceed)
            - bb_width: Bollinger Band Width (normalized)
            - rsi_14: 14-period RSI
            - volume_ratio: Current volume / 20d average volume
            - deviation_z: Z-score of price deviation from SMA
            - signal_strength: Combined signal strength metric
        """
        prices = data.get("close")
        volumes = data.get("volume")

        if prices is None or prices.empty:
            raise ValueError("Price data ('close') is required")
        if len(prices) < self._params["bb_window"] + 10:
            raise ValueError(f"Need at least {self._params['bb_window'] + 10} days of data")

        bb_window = self._params["bb_window"]
        bb_std = self._params["bb_std"]
        rsi_period = self._params["rsi_period"]

        features_dict = {}

        for ticker in prices.columns:
            p = prices[ticker].dropna()
            if len(p) < bb_window + rsi_period:
                continue

            # Bollinger Bands
            sma = p.rolling(bb_window).mean()
            std = p.rolling(bb_window).std()
            upper = sma + bb_std * std
            lower = sma - bb_std * std

            last_price = p.iloc[-1]
            last_sma = sma.iloc[-1]
            last_upper = upper.iloc[-1]
            last_lower = lower.iloc[-1]
            last_std = std.iloc[-1]

            # %B
            band_range = last_upper - last_lower
            pct_b = (last_price - last_lower) / band_range if band_range > 0 else 0.5

            # Bollinger Band Width
            bbw = band_range / last_sma if last_sma > 0 else 0

            # RSI
            rsi = self._compute_rsi(p, rsi_period)

            # Z-score deviation from SMA
            deviation_z = (last_price - last_sma) / last_std if last_std > 0 else 0

            # Volume ratio
            vol_ratio = 1.0
            if volumes is not None and ticker in volumes.columns:
                v = volumes[ticker].dropna()
                if len(v) >= 20:
                    avg_vol = v.iloc[-20:].mean()
                    vol_ratio = v.iloc[-1] / avg_vol if avg_vol > 0 else 1.0

            # Signal strength: how far from mean + RSI extremity
            signal_strength = abs(deviation_z) * (1 + abs(rsi - 50) / 50)

            features_dict[ticker] = {
                "last_price": last_price,
                "sma_20": last_sma,
                "bb_upper": last_upper,
                "bb_lower": last_lower,
                "pct_b": pct_b,
                "bb_width": bbw,
                "rsi_14": rsi,
                "deviation_z": deviation_z,
                "volume_ratio": vol_ratio,
                "signal_strength": signal_strength,
            }

        features = pd.DataFrame(features_dict).T

        # Add volume data for ADV filter
        if volumes is not None:
            adv_20 = volumes.iloc[-20:].mean()
            features["avg_dollar_volume"] = (adv_20 * prices.iloc[-1]).reindex(features.index)

        # Add sectors if available
        sectors = data.get("sectors")
        if sectors is not None:
            features["sector"] = sectors.reindex(features.index)

        return features

    def _compute_rsi(self, prices: pd.Series, period: int) -> float:
        """Compute RSI for the latest data point."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)

        avg_gain = gain.rolling(period).mean().iloc[-1]
        avg_loss = loss.rolling(period).mean().iloc[-1]

        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def generate_signal(self, features: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Generate mean reversion signal based on Bollinger Bands + RSI.

        Process:
            1. Filter by price, ADV, and bandwidth
            2. Identify oversold (long) and overbought (short) candidates
            3. Apply RSI confirmation if enabled
            4. Apply volume confirmation if enabled
            5. Weight by signal strength (deviation magnitude)
            6. Cap at max_positions
        """
        min_price = params.get("min_price", 10.0)
        min_adv = params.get("min_adv", 5_000_000)
        min_bbw = params.get("min_bbw", 0.02)
        max_bbw = params.get("max_bbw", 0.30)
        pct_b_long = params.get("pct_b_long_threshold", 0.05)
        pct_b_short = params.get("pct_b_short_threshold", 0.95)
        use_rsi = params.get("use_rsi_confirmation", True)
        rsi_os = params.get("rsi_oversold", 30)
        rsi_ob = params.get("rsi_overbought", 70)
        use_vol = params.get("volume_confirmation", True)
        vol_threshold = params.get("volume_ratio_threshold", 1.2)
        max_positions = params.get("max_positions", 20)

        signal = pd.Series(0.0, index=features.index)

        if len(features) == 0:
            return signal

        # Apply filters
        mask = pd.Series(True, index=features.index)
        if "last_price" in features.columns:
            mask &= features["last_price"] >= min_price
        if "avg_dollar_volume" in features.columns:
            mask &= features["avg_dollar_volume"] >= min_adv
        if "bb_width" in features.columns:
            mask &= features["bb_width"] >= min_bbw
            mask &= features["bb_width"] <= max_bbw

        filtered = features[mask]

        if len(filtered) == 0:
            return signal

        # Identify long candidates (oversold)
        long_mask = filtered["pct_b"] < pct_b_long
        if use_rsi:
            long_mask &= filtered["rsi_14"] < rsi_os
        if use_vol:
            long_mask &= filtered["volume_ratio"] >= vol_threshold

        # Identify short candidates (overbought)
        short_mask = filtered["pct_b"] > pct_b_short
        if use_rsi:
            short_mask &= filtered["rsi_14"] > rsi_ob
        if use_vol:
            short_mask &= filtered["volume_ratio"] >= vol_threshold

        long_candidates = filtered[long_mask]
        short_candidates = filtered[short_mask]

        # Weight by signal strength (deviation from mean)
        if len(long_candidates) > 0:
            strengths = long_candidates["signal_strength"]
            # Sort by strength, take top max_positions/2
            top_longs = strengths.sort_values(ascending=False).head(max_positions // 2)
            if top_longs.sum() > 0:
                weights = top_longs / top_longs.sum()
                signal[weights.index] = weights

        if len(short_candidates) > 0:
            strengths = short_candidates["signal_strength"]
            top_shorts = strengths.sort_values(ascending=False).head(max_positions // 2)
            if top_shorts.sum() > 0:
                weights = top_shorts / top_shorts.sum()
                signal[weights.index] = -weights

        return signal

    def check_risk(self, targets: pd.Series, risk_context: Dict[str, Any]) -> RiskCheckResult:
        """Enhanced risk check with VIX regime filter and position count limit."""
        result = super().check_risk(targets, risk_context)

        # VIX regime filter
        vix = risk_context.get("vix_level", 0)
        max_vix = self._params.get("max_vix", 35)
        if vix > max_vix:
            result.passed = False
            result.hard_breaches.append(
                f"VIX {vix:.1f} > {max_vix}: mean reversion signals suppressed in crisis regime"
            )

        # Max positions check
        max_pos = self._params.get("max_positions", 20)
        n_positions = (targets != 0).sum()
        if n_positions > max_pos:
            result.soft_warnings.append(
                f"Position count {n_positions} exceeds limit {max_pos}"
            )

        return result
