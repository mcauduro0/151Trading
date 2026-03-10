"""
EQ_MOM_001 — Cross-Sectional Momentum Strategy

Ranks US large-cap equities by 12-1 month momentum (Jegadeesh & Titman, 1993).
Goes long the top decile and short the bottom decile of the cross-section.
Rebalances monthly with sector-neutrality constraints.

Signal Construction:
    momentum_score = (P_t-21 / P_t-252) - 1   [skip most recent month]
    z_score = (momentum_score - mean) / std     [cross-sectional standardization]
    signal = z_score clipped to [-3, +3]

Position Sizing:
    - Equal-weight within deciles (default)
    - Optional: signal-weighted proportional
    - Sector-neutral: zero net exposure per GICS sector

Risk Controls:
    - Max single-name weight: 5%
    - Max sector tilt: 10% vs benchmark
    - Turnover limit: 200% annualized
    - Skip rebalance if VIX > 40 (crash regime)
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


class CrossSectionalMomentum(BaseStrategy):
    """
    Cross-Sectional Momentum (12-1 month).

    Academic basis: Jegadeesh & Titman (1993), Asness et al. (2013)
    The strategy exploits the well-documented tendency for stocks that have
    performed well over the past 12 months (excluding the most recent month)
    to continue outperforming, and vice versa.
    """

    DEFAULT_PARAMS = {
        "lookback_days": 252,       # ~12 months of trading days
        "skip_days": 21,            # Skip most recent month (reversal effect)
        "top_pct": 0.10,            # Long top 10%
        "bottom_pct": 0.10,         # Short bottom 10%
        "book_size": 1_000_000,
        "max_single_weight": 0.05,
        "sector_neutral": True,
        "min_price": 5.0,           # Minimum price filter
        "min_volume_avg": 1_000_000, # Minimum 20-day avg volume
        "vix_crash_threshold": 40,  # Skip rebalance if VIX > this
        "max_turnover_annual": 2.0, # 200% annualized turnover limit
        "weighting": "equal",       # "equal" or "signal_weighted"
    }

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        merged = {**self.DEFAULT_PARAMS, **(params or {})}
        super().__init__(merged)

    def get_metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            code="EQ_MOM_001",
            name="Cross-Sectional Momentum",
            source_book="151TS",
            asset_class=AssetClass.EQUITY,
            style=StrategyStyle.MOMENTUM,
            sub_style="12-1 month cross-sectional",
            horizon="monthly",
            directionality="long_short",
            complexity="moderate",
            description=(
                "Ranks US large-cap equities by 12-1 month momentum. "
                "Goes long top decile, short bottom decile. "
                "Monthly rebalance with sector-neutrality constraints."
            ),
            math_formula="mom_score = P(t-21)/P(t-252) - 1; z = (mom - μ) / σ",
            assumptions=[
                "Momentum premium persists in large-cap US equities",
                "1-month skip avoids short-term reversal contamination",
                "Sector neutrality reduces crash risk during momentum reversals",
            ],
            known_failure_modes=[
                "Momentum crashes during sharp market reversals (2009 Q1)",
                "Crowding risk when momentum factor is popular",
                "Underperforms in range-bound, low-dispersion markets",
            ],
            capacity_notes="High capacity in large-cap (>$50M daily volume per name)",
            required_data=["bars_1d", "sector_classification", "vix"],
            parameters=self._params,
            parameter_bounds={
                "lookback_days": (126, 504),
                "skip_days": (0, 63),
                "top_pct": (0.05, 0.30),
                "bottom_pct": (0.05, 0.30),
                "max_single_weight": (0.01, 0.10),
                "vix_crash_threshold": (25, 60),
            },
        )

    def generate_features(self, context: StrategyContext, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Compute momentum features from price data.

        Features generated:
            - mom_12_1: 12-month return excluding last month
            - mom_z_score: Cross-sectional z-score of momentum
            - avg_volume_20d: 20-day average volume
            - last_price: Most recent closing price
            - sector: GICS sector (if available)
        """
        prices = data.get("close")
        volumes = data.get("volume")

        if prices is None or prices.empty:
            raise ValueError("Price data ('close') is required for momentum calculation")

        lookback = self._params["lookback_days"]
        skip = self._params["skip_days"]

        # Ensure we have enough data
        if len(prices) < lookback + skip + 10:
            raise ValueError(
                f"Need at least {lookback + skip + 10} days of data, got {len(prices)}"
            )

        # Compute 12-1 month momentum: return from t-252 to t-21
        price_end = prices.iloc[-skip - 1] if skip > 0 else prices.iloc[-1]
        price_start = prices.iloc[-(lookback + skip)]

        mom_12_1 = (price_end / price_start) - 1

        # Drop NaN and inf
        mom_12_1 = mom_12_1.replace([np.inf, -np.inf], np.nan).dropna()

        # Cross-sectional z-score
        mom_mean = mom_12_1.mean()
        mom_std = mom_12_1.std()
        if mom_std > 0:
            mom_z = (mom_12_1 - mom_mean) / mom_std
        else:
            mom_z = pd.Series(0, index=mom_12_1.index)

        # Clip z-scores to [-3, +3]
        mom_z = mom_z.clip(-3, 3)

        # Volume filter
        avg_vol_20d = volumes.iloc[-20:].mean() if volumes is not None else pd.Series(np.nan, index=mom_12_1.index)

        # Last price
        last_price = prices.iloc[-1]

        # Build feature DataFrame
        features = pd.DataFrame({
            "mom_12_1": mom_12_1,
            "mom_z_score": mom_z,
            "avg_volume_20d": avg_vol_20d,
            "last_price": last_price,
        })

        # Add sector data if available
        sectors = data.get("sectors")
        if sectors is not None:
            features["sector"] = sectors

        return features

    def generate_signal(self, features: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Generate long/short signal from momentum features.

        Process:
            1. Filter by price and volume
            2. Rank by momentum z-score
            3. Long top decile, short bottom decile
            4. Apply sector-neutral adjustment (optional)
        """
        min_price = params.get("min_price", 5.0)
        min_volume = params.get("min_volume_avg", 1_000_000)
        top_pct = params.get("top_pct", 0.10)
        bottom_pct = params.get("bottom_pct", 0.10)
        weighting = params.get("weighting", "equal")
        sector_neutral = params.get("sector_neutral", True)

        # Apply filters
        mask = pd.Series(True, index=features.index)
        if "last_price" in features.columns:
            mask &= features["last_price"] >= min_price
        if "avg_volume_20d" in features.columns:
            mask &= features["avg_volume_20d"] >= min_volume

        filtered = features[mask].copy()

        if len(filtered) < 20:
            return pd.Series(0, index=features.index, dtype=float)

        # Rank by momentum z-score
        z_scores = filtered["mom_z_score"]
        n = len(z_scores)
        n_long = max(int(n * top_pct), 1)
        n_short = max(int(n * bottom_pct), 1)

        ranked = z_scores.sort_values(ascending=False)
        long_names = ranked.head(n_long).index
        short_names = ranked.tail(n_short).index

        # Build signal
        signal = pd.Series(0.0, index=features.index)

        if weighting == "equal":
            signal[long_names] = 1.0 / n_long
            signal[short_names] = -1.0 / n_short
        else:  # signal_weighted
            long_z = z_scores[long_names]
            short_z = z_scores[short_names]
            signal[long_names] = long_z / long_z.abs().sum() if long_z.abs().sum() > 0 else 1.0 / n_long
            signal[short_names] = short_z / short_z.abs().sum() if short_z.abs().sum() > 0 else -1.0 / n_short

        # Sector-neutral adjustment
        if sector_neutral and "sector" in filtered.columns:
            signal = self._sector_neutralize(signal, filtered["sector"])

        return signal

    def _sector_neutralize(self, signal: pd.Series, sectors: pd.Series) -> pd.Series:
        """
        Adjust signal to be sector-neutral.
        Within each sector, demean the signal so net exposure is zero.
        """
        adjusted = signal.copy()
        for sector in sectors.unique():
            if pd.isna(sector):
                continue
            sector_mask = sectors == sector
            sector_signal = signal[sector_mask]
            if len(sector_signal) > 1:
                sector_mean = sector_signal.mean()
                adjusted[sector_mask] = sector_signal - sector_mean

        # Re-normalize to sum to ~0 (market neutral)
        long_sum = adjusted[adjusted > 0].sum()
        short_sum = adjusted[adjusted < 0].sum()
        if long_sum > 0 and short_sum < 0:
            adjusted[adjusted > 0] /= long_sum
            adjusted[adjusted < 0] /= abs(short_sum)

        return adjusted

    def check_risk(self, targets: pd.Series, risk_context: Dict[str, Any]) -> RiskCheckResult:
        """Enhanced risk check with VIX crash regime detection."""
        result = super().check_risk(targets, risk_context)

        # VIX crash regime check
        vix = risk_context.get("vix_level")
        threshold = self._params.get("vix_crash_threshold", 40)
        if vix is not None and vix > threshold:
            result.hard_breaches.append(
                f"VIX at {vix:.1f} exceeds crash threshold {threshold}. "
                "Skipping rebalance to avoid momentum crash risk."
            )
            result.passed = False

        return result
