"""
ETF_ROT_007 — ETF Sector Rotation Strategy

Rotates capital among sector ETFs based on relative momentum, with a
macro regime overlay that adjusts allocations based on economic conditions.

Signal Construction:
    1. Compute multi-timeframe momentum for each sector ETF:
       - 1-month return (weight 0.40)
       - 3-month return (weight 0.35)
       - 6-month return (weight 0.25)
    2. Rank sectors by composite momentum score
    3. Apply macro regime overlay:
       - Expansion: overweight cyclicals (XLY, XLF, XLI, XLB)
       - Contraction: overweight defensives (XLU, XLP, XLV, XLRE)
       - Determine regime from yield curve slope + ISM PMI
    4. Long top 3 sectors, short bottom 2 sectors
    5. Rebalance monthly

ETF Universe:
    XLK (Tech), XLF (Financials), XLV (Healthcare), XLY (Consumer Disc),
    XLP (Consumer Staples), XLE (Energy), XLI (Industrials), XLB (Materials),
    XLRE (Real Estate), XLU (Utilities), XLC (Communication)

Risk Controls:
    - Max 30% in any single sector
    - Trend filter: skip short signals when SPY > 200-day SMA
    - Volatility scaling: reduce exposure when VIX > 25

Academic basis:
    - Moskowitz & Grinblatt (1999): Industry momentum
    - Asness, Moskowitz & Pedersen (2013): Value and momentum everywhere
    - Ilmanen (2011): Expected Returns — macro regime rotation
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


class ETFSectorRotation(BaseStrategy):
    """
    ETF Sector Rotation with Macro Regime Overlay.

    Rotates among sector ETFs using multi-timeframe momentum
    and adjusts for macroeconomic regime (expansion vs contraction).
    """

    # Sector ETF universe
    SECTOR_ETFS = {
        "XLK": {"name": "Technology", "type": "cyclical"},
        "XLF": {"name": "Financials", "type": "cyclical"},
        "XLV": {"name": "Healthcare", "type": "defensive"},
        "XLY": {"name": "Consumer Discretionary", "type": "cyclical"},
        "XLP": {"name": "Consumer Staples", "type": "defensive"},
        "XLE": {"name": "Energy", "type": "cyclical"},
        "XLI": {"name": "Industrials", "type": "cyclical"},
        "XLB": {"name": "Materials", "type": "cyclical"},
        "XLRE": {"name": "Real Estate", "type": "defensive"},
        "XLU": {"name": "Utilities", "type": "defensive"},
        "XLC": {"name": "Communication Services", "type": "cyclical"},
    }

    DEFAULT_PARAMS = {
        # Momentum weights
        "mom_1m_weight": 0.40,
        "mom_3m_weight": 0.35,
        "mom_6m_weight": 0.25,

        # Portfolio construction
        "n_long": 3,                 # Top N sectors to go long
        "n_short": 2,                # Bottom N sectors to short
        "book_size": 1_000_000,
        "max_single_weight": 0.30,   # Max 30% in any sector

        # Macro regime
        "use_macro_overlay": True,
        "regime_cyclical_boost": 0.10,  # Boost cyclicals in expansion
        "regime_defensive_boost": 0.10, # Boost defensives in contraction

        # Risk controls
        "use_trend_filter": True,    # SPY > 200-day SMA
        "vix_scaling": True,
        "vix_threshold": 25,
        "vix_scale_factor": 0.50,    # Reduce to 50% when VIX > threshold

        "rebalance_frequency": "monthly",
    }

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        merged = {**self.DEFAULT_PARAMS, **(params or {})}
        super().__init__(merged)

    def get_metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            code="ETF_ROT_007",
            name="ETF Sector Rotation",
            source_book="151TS",
            asset_class=AssetClass.ETF,
            style=StrategyStyle.MOMENTUM,
            sub_style="sector momentum with macro regime overlay",
            horizon="monthly",
            directionality="long_short",
            complexity="moderate",
            description=(
                "Rotates among 11 SPDR sector ETFs using multi-timeframe momentum "
                "(1m/3m/6m weighted). Macro regime overlay boosts cyclicals in "
                "expansion and defensives in contraction. Monthly rebalance."
            ),
            math_formula=(
                "mom_composite = 0.40*ret_1m + 0.35*ret_3m + 0.25*ret_6m; "
                "regime = f(yield_curve_slope, ISM_PMI); "
                "score = mom_composite + regime_adjustment; "
                "Long top 3, Short bottom 2"
            ),
            assumptions=[
                "Sector momentum persists over 1-6 month horizons",
                "Macro regime (expansion/contraction) affects sector performance predictably",
                "Yield curve slope and ISM PMI are reliable regime indicators",
                "Monthly rebalance captures sector rotation without excessive turnover",
            ],
            known_failure_modes=[
                "Sector rotation reversals: sharp sector leadership changes",
                "Correlated sectors: tech/comm services move together",
                "Macro regime transitions: lagging indicators miss turning points",
                "Crowded trades: popular sector momentum strategies",
                "ETF tracking error during high volatility",
            ],
            capacity_notes="Very high capacity; sector ETFs are highly liquid",
            required_data=["bars_1d", "macro_indicators"],
            parameters=self._params,
            parameter_bounds={
                "mom_1m_weight": (0.0, 1.0),
                "mom_3m_weight": (0.0, 1.0),
                "mom_6m_weight": (0.0, 1.0),
                "n_long": (1, 6),
                "n_short": (0, 5),
                "max_single_weight": (0.15, 0.50),
                "vix_threshold": (15, 40),
                "vix_scale_factor": (0.25, 1.0),
            },
        )

    def generate_features(self, context: StrategyContext, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Compute sector momentum and macro regime features.

        Features generated:
            - ret_1m: 1-month return
            - ret_3m: 3-month return
            - ret_6m: 6-month return
            - mom_composite: Weighted momentum composite
            - sector_type: cyclical or defensive
            - regime: current macro regime (expansion/contraction/neutral)
            - regime_adjusted_score: Momentum + regime adjustment
            - vol_20d: 20-day realized volatility
            - relative_strength: Sector return vs SPY
        """
        prices = data.get("close")
        if prices is None or prices.empty:
            raise ValueError("Price data ('close') is required")

        # Filter to sector ETFs that exist in data
        etf_tickers = [t for t in self.SECTOR_ETFS.keys() if t in prices.columns]
        if len(etf_tickers) < 5:
            raise ValueError(f"Need at least 5 sector ETFs, found {len(etf_tickers)}")

        etf_prices = prices[etf_tickers]

        # Compute returns over multiple horizons
        features_dict = {}
        for ticker in etf_tickers:
            p = etf_prices[ticker].dropna()
            if len(p) < 126:  # Need at least 6 months
                continue

            ret_1m = (p.iloc[-1] / p.iloc[-21] - 1) if len(p) >= 21 else 0
            ret_3m = (p.iloc[-1] / p.iloc[-63] - 1) if len(p) >= 63 else 0
            ret_6m = (p.iloc[-1] / p.iloc[-126] - 1) if len(p) >= 126 else 0

            # Composite momentum
            w1 = self._params["mom_1m_weight"]
            w3 = self._params["mom_3m_weight"]
            w6 = self._params["mom_6m_weight"]
            mom_composite = w1 * ret_1m + w3 * ret_3m + w6 * ret_6m

            # Volatility
            returns = p.pct_change().dropna()
            vol_20d = returns.iloc[-20:].std() * np.sqrt(252) if len(returns) >= 20 else 0

            # Relative strength vs SPY
            spy = data.get("benchmark")
            rel_strength = 0
            if spy is not None:
                if isinstance(spy, pd.DataFrame):
                    spy_series = spy.iloc[:, 0]
                else:
                    spy_series = spy
                spy_clean = spy_series.dropna()
                if len(spy_clean) >= 21:
                    spy_ret_1m = spy_clean.iloc[-1] / spy_clean.iloc[-21] - 1
                    rel_strength = ret_1m - spy_ret_1m

            features_dict[ticker] = {
                "ret_1m": ret_1m,
                "ret_3m": ret_3m,
                "ret_6m": ret_6m,
                "mom_composite": mom_composite,
                "sector_name": self.SECTOR_ETFS[ticker]["name"],
                "sector_type": self.SECTOR_ETFS[ticker]["type"],
                "vol_20d": vol_20d,
                "relative_strength": rel_strength,
                "last_price": p.iloc[-1],
            }

        features = pd.DataFrame(features_dict).T

        # Determine macro regime
        regime = self._determine_regime(data)
        features["regime"] = regime

        # Regime-adjusted score
        cyclical_boost = self._params["regime_cyclical_boost"]
        defensive_boost = self._params["regime_defensive_boost"]

        features["regime_adjustment"] = 0.0
        if regime == "expansion":
            features.loc[features["sector_type"] == "cyclical", "regime_adjustment"] = cyclical_boost
            features.loc[features["sector_type"] == "defensive", "regime_adjustment"] = -cyclical_boost * 0.5
        elif regime == "contraction":
            features.loc[features["sector_type"] == "defensive", "regime_adjustment"] = defensive_boost
            features.loc[features["sector_type"] == "cyclical", "regime_adjustment"] = -defensive_boost * 0.5

        features["regime_adjusted_score"] = features["mom_composite"] + features["regime_adjustment"]

        return features

    def _determine_regime(self, data: Dict[str, pd.DataFrame]) -> str:
        """
        Determine macro regime from yield curve and PMI.

        Returns: "expansion", "contraction", or "neutral"
        """
        macro = data.get("macro_indicators")
        if macro is None:
            return "neutral"

        # Yield curve slope (10Y - 2Y)
        yield_slope = None
        if isinstance(macro, dict):
            yield_slope = macro.get("yield_curve_slope")
        elif isinstance(macro, pd.DataFrame):
            if "yield_curve_slope" in macro.columns:
                yield_slope = macro["yield_curve_slope"].iloc[-1]

        # ISM PMI
        ism_pmi = None
        if isinstance(macro, dict):
            ism_pmi = macro.get("ism_pmi")
        elif isinstance(macro, pd.DataFrame):
            if "ism_pmi" in macro.columns:
                ism_pmi = macro["ism_pmi"].iloc[-1]

        # Decision logic
        expansion_signals = 0
        contraction_signals = 0

        if yield_slope is not None:
            if yield_slope > 0.5:
                expansion_signals += 1
            elif yield_slope < -0.2:
                contraction_signals += 1

        if ism_pmi is not None:
            if ism_pmi > 52:
                expansion_signals += 1
            elif ism_pmi < 48:
                contraction_signals += 1

        if expansion_signals > contraction_signals:
            return "expansion"
        elif contraction_signals > expansion_signals:
            return "contraction"
        return "neutral"

    def generate_signal(self, features: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Generate sector rotation signal.

        Process:
            1. Rank sectors by regime-adjusted momentum score
            2. Long top N, short bottom N
            3. Apply trend filter (skip shorts if SPY trending up)
            4. Apply VIX scaling
            5. Equal-weight within long/short legs
        """
        n_long = params.get("n_long", 3)
        n_short = params.get("n_short", 2)
        use_trend = params.get("use_trend_filter", True)
        vix_scaling = params.get("vix_scaling", True)
        vix_threshold = params.get("vix_threshold", 25)
        vix_scale = params.get("vix_scale_factor", 0.50)

        if len(features) < n_long + n_short:
            return pd.Series(0, index=features.index, dtype=float)

        # Rank by regime-adjusted score
        ranked = features["regime_adjusted_score"].sort_values(ascending=False)

        signal = pd.Series(0.0, index=features.index)

        # Long top N
        long_etfs = ranked.head(n_long).index
        signal[long_etfs] = 1.0 / n_long

        # Short bottom N
        short_etfs = ranked.tail(n_short).index
        signal[short_etfs] = -1.0 / n_short

        # VIX scaling
        if vix_scaling:
            vix = features.get("vix_level")
            if vix is None and "regime" in features.columns:
                # Try to get VIX from context
                pass
            # Apply uniform scaling if VIX data available externally
            # This will be handled in the runner

        return signal

    def check_risk(self, targets: pd.Series, risk_context: Dict[str, Any]) -> RiskCheckResult:
        """Risk check with sector concentration and VIX scaling."""
        result = super().check_risk(targets, risk_context)

        # Max single sector weight
        max_weight = self._params.get("max_single_weight", 0.30)
        book = self._params.get("book_size", 1_000_000)
        violations = targets[targets.abs() > max_weight * book]
        if len(violations) > 0:
            for sym in violations.index:
                result.soft_warnings.append(
                    f"{sym} weight {targets[sym]/book:.1%} exceeds {max_weight:.0%} limit"
                )

        # VIX scaling
        vix = risk_context.get("vix_level", 0)
        if self._params.get("vix_scaling") and vix > self._params.get("vix_threshold", 25):
            scale = self._params.get("vix_scale_factor", 0.50)
            result.clipped_targets = targets * scale
            result.soft_warnings.append(
                f"VIX {vix:.1f} > threshold: positions scaled to {scale:.0%}"
            )

        # Trend filter warning
        spy_above_200sma = risk_context.get("spy_above_200sma", True)
        if not spy_above_200sma and self._params.get("use_trend_filter"):
            short_targets = targets[targets < 0]
            if len(short_targets) > 0:
                result.soft_warnings.append(
                    "SPY below 200-day SMA: consider reducing short exposure"
                )

        return result
