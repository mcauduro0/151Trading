"""
EQ_LVOL_003 — Low Volatility Anomaly Strategy

Exploits the empirical finding that low-beta and low-volatility stocks
deliver higher risk-adjusted returns than high-beta stocks (the "betting
against beta" anomaly).

Signal Construction:
    1. Compute realized volatility (60-day rolling std of daily returns)
    2. Compute CAPM beta vs SPY (252-day rolling regression)
    3. Composite: low_vol_score = -0.50 * z(vol) - 0.50 * z(beta)
    4. Long bottom quintile (lowest vol/beta), short top quintile

Position Sizing:
    - Inverse-volatility weighted within each leg
    - Max single-name weight: 5%

Risk Controls:
    - Sector concentration limit: 25% per sector
    - Beta-adjusted: target portfolio beta ~0
    - Leverage limit: 2x gross

Academic basis:
    - Frazzini & Pedersen (2014): Betting Against Beta
    - Baker, Bradley & Wurgler (2011): Benchmarks as Limits to Arbitrage
    - Ang, Hodrick, Xing & Zhang (2006): High idiosyncratic vol → low returns
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


class LowVolatilityAnomaly(BaseStrategy):
    """
    Low Volatility Anomaly (Betting Against Beta).

    Exploits the well-documented anomaly that low-volatility and low-beta
    stocks deliver superior risk-adjusted returns compared to their
    high-volatility counterparts.
    """

    DEFAULT_PARAMS = {
        "vol_window": 60,           # Rolling volatility window (days)
        "beta_window": 252,         # Rolling beta window (days)
        "weight_vol": 0.50,         # Weight on volatility component
        "weight_beta": 0.50,        # Weight on beta component
        "top_pct": 0.20,            # Short top 20% (highest vol/beta)
        "bottom_pct": 0.20,         # Long bottom 20% (lowest vol/beta)
        "book_size": 1_000_000,
        "max_single_weight": 0.05,
        "max_sector_weight": 0.25,  # Max 25% in any sector
        "target_beta": 0.0,         # Target portfolio beta
        "max_gross_leverage": 2.0,
        "min_price": 10.0,
        "min_volume_avg": 2_000_000,
        "inverse_vol_weighting": True,
        "rebalance_frequency": "monthly",
    }

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        merged = {**self.DEFAULT_PARAMS, **(params or {})}
        super().__init__(merged)

    def get_metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            code="EQ_LVOL_003",
            name="Low Volatility Anomaly",
            source_book="151TS",
            asset_class=AssetClass.EQUITY,
            style=StrategyStyle.FACTOR,
            sub_style="betting against beta / low volatility",
            horizon="monthly",
            directionality="long_short",
            complexity="moderate",
            description=(
                "Exploits the low-volatility anomaly by going long low-beta/low-vol "
                "stocks and short high-beta/high-vol stocks. Inverse-volatility weighted "
                "with sector constraints. Monthly rebalance."
            ),
            math_formula=(
                "vol_60d = std(r, 60); beta_252d = cov(r, r_mkt)/var(r_mkt); "
                "score = -0.5*z(vol) - 0.5*z(beta)"
            ),
            assumptions=[
                "Low-volatility anomaly persists due to leverage constraints and lottery preferences",
                "CAPM beta is a meaningful risk measure for cross-sectional ranking",
                "Inverse-vol weighting improves risk-adjusted returns",
                "Monthly rebalance captures slow-moving vol regime changes",
            ],
            known_failure_modes=[
                "Underperforms in strong bull markets (low-beta drag)",
                "Sector concentration in utilities/staples during risk-off periods",
                "Crowding risk as low-vol strategies have become popular",
                "Sudden vol regime shifts can cause large tracking error",
            ],
            capacity_notes="Very high capacity in large-cap equities",
            required_data=["bars_1d", "market_benchmark", "sector_classification"],
            parameters=self._params,
            parameter_bounds={
                "vol_window": (20, 126),
                "beta_window": (126, 504),
                "weight_vol": (0.0, 1.0),
                "weight_beta": (0.0, 1.0),
                "top_pct": (0.10, 0.40),
                "bottom_pct": (0.10, 0.40),
                "max_sector_weight": (0.10, 0.50),
            },
        )

    def generate_features(self, context: StrategyContext, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Compute volatility and beta features.

        Features generated:
            - realized_vol_60d: 60-day rolling annualized volatility
            - beta_252d: 252-day rolling CAPM beta vs SPY
            - idio_vol: Idiosyncratic volatility (residual from CAPM regression)
            - vol_z_score: Cross-sectional z-score of volatility
            - beta_z_score: Cross-sectional z-score of beta
            - low_vol_composite: Weighted composite of vol and beta z-scores
        """
        prices = data.get("close")
        benchmark = data.get("benchmark")  # SPY prices

        if prices is None or prices.empty:
            raise ValueError("Price data ('close') is required")

        vol_window = self._params["vol_window"]
        beta_window = self._params["beta_window"]

        # Daily returns
        returns = prices.pct_change().dropna(how="all")

        # Benchmark returns
        if benchmark is not None and not benchmark.empty:
            if isinstance(benchmark, pd.DataFrame):
                bench_ret = benchmark.iloc[:, 0].pct_change().dropna()
            else:
                bench_ret = benchmark.pct_change().dropna()
        else:
            # Use equal-weighted market return as proxy
            bench_ret = returns.mean(axis=1)

        # 1. Realized volatility (annualized)
        realized_vol = returns.iloc[-vol_window:].std() * np.sqrt(252)

        # 2. Rolling CAPM beta
        betas = pd.Series(index=returns.columns, dtype=float)
        idio_vols = pd.Series(index=returns.columns, dtype=float)

        recent_returns = returns.iloc[-beta_window:]
        recent_bench = bench_ret.iloc[-beta_window:]

        # Align indices
        common_idx = recent_returns.index.intersection(recent_bench.index)
        recent_returns = recent_returns.loc[common_idx]
        recent_bench = recent_bench.loc[common_idx]

        bench_var = recent_bench.var()

        for col in returns.columns:
            stock_ret = recent_returns[col].dropna()
            common = stock_ret.index.intersection(recent_bench.index)
            if len(common) < beta_window * 0.5:
                betas[col] = np.nan
                idio_vols[col] = np.nan
                continue

            sr = stock_ret.loc[common]
            br = recent_bench.loc[common]

            cov = sr.cov(br)
            if bench_var > 0:
                beta = cov / bench_var
            else:
                beta = 1.0

            betas[col] = beta

            # Idiosyncratic volatility: std of residuals
            residuals = sr - beta * br
            idio_vols[col] = residuals.std() * np.sqrt(252)

        # Cross-sectional z-scores
        vol_clean = realized_vol.replace([np.inf, -np.inf], np.nan).dropna()
        beta_clean = betas.replace([np.inf, -np.inf], np.nan).dropna()

        vol_z = self._zscore(vol_clean)
        beta_z = self._zscore(beta_clean)

        # Composite: negative because we want LOW vol/beta to score HIGH
        w_vol = self._params["weight_vol"]
        w_beta = self._params["weight_beta"]

        features = pd.DataFrame({
            "realized_vol_60d": realized_vol,
            "beta_252d": betas,
            "idio_vol": idio_vols,
            "vol_z_score": vol_z,
            "beta_z_score": beta_z,
            "low_vol_composite": -(w_vol * vol_z.reindex(prices.columns, fill_value=0) +
                                    w_beta * beta_z.reindex(prices.columns, fill_value=0)),
            "last_price": prices.iloc[-1],
        })

        # Volume
        volumes = data.get("volume")
        if volumes is not None:
            features["avg_volume_20d"] = volumes.iloc[-20:].mean()

        # Sectors
        sectors = data.get("sectors")
        if sectors is not None:
            features["sector"] = sectors

        return features

    def _zscore(self, series: pd.Series) -> pd.Series:
        """Cross-sectional z-score with winsorization."""
        if len(series) < 5 or series.std() == 0:
            return pd.Series(0, index=series.index)
        lower = series.quantile(0.01)
        upper = series.quantile(0.99)
        winsorized = series.clip(lower, upper)
        return (winsorized - winsorized.mean()) / winsorized.std()

    def generate_signal(self, features: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Generate low-vol signal.

        Process:
            1. Filter by price and volume
            2. Rank by low_vol_composite (high = low vol/beta)
            3. Long bottom quintile, short top quintile
            4. Inverse-vol weight within each leg
            5. Apply sector concentration limits
        """
        min_price = params.get("min_price", 10.0)
        min_volume = params.get("min_volume_avg", 2_000_000)
        top_pct = params.get("top_pct", 0.20)
        bottom_pct = params.get("bottom_pct", 0.20)
        inv_vol = params.get("inverse_vol_weighting", True)
        max_sector = params.get("max_sector_weight", 0.25)

        # Filters
        mask = pd.Series(True, index=features.index)
        if "last_price" in features.columns:
            mask &= features["last_price"] >= min_price
        if "avg_volume_20d" in features.columns:
            mask &= features["avg_volume_20d"] >= min_volume

        filtered = features[mask].dropna(subset=["low_vol_composite"])

        if len(filtered) < 20:
            return pd.Series(0, index=features.index, dtype=float)

        # Rank by composite (higher = lower vol/beta = more desirable)
        composite = filtered["low_vol_composite"]
        n = len(composite)
        n_long = max(int(n * bottom_pct), 1)
        n_short = max(int(n * top_pct), 1)

        ranked = composite.sort_values(ascending=False)
        long_names = ranked.head(n_long).index
        short_names = ranked.tail(n_short).index

        # Build signal with inverse-vol weighting
        signal = pd.Series(0.0, index=features.index)

        if inv_vol and "realized_vol_60d" in features.columns:
            # Inverse-vol weights for longs
            long_vols = features.loc[long_names, "realized_vol_60d"].replace(0, np.nan).dropna()
            if len(long_vols) > 0:
                inv_weights = 1.0 / long_vols
                inv_weights /= inv_weights.sum()
                signal[inv_weights.index] = inv_weights

            # Inverse-vol weights for shorts
            short_vols = features.loc[short_names, "realized_vol_60d"].replace(0, np.nan).dropna()
            if len(short_vols) > 0:
                inv_weights = 1.0 / short_vols
                inv_weights /= inv_weights.sum()
                signal[inv_weights.index] = -inv_weights
        else:
            signal[long_names] = 1.0 / n_long
            signal[short_names] = -1.0 / n_short

        # Sector concentration limits
        if "sector" in features.columns:
            signal = self._apply_sector_limits(signal, features["sector"], max_sector)

        return signal

    def _apply_sector_limits(self, signal: pd.Series, sectors: pd.Series, max_weight: float) -> pd.Series:
        """Clip sector weights to max_weight and redistribute."""
        adjusted = signal.copy()

        for sector in sectors.dropna().unique():
            sector_mask = sectors == sector
            sector_signal = signal[sector_mask]
            sector_long = sector_signal[sector_signal > 0].sum()
            sector_short = abs(sector_signal[sector_signal < 0].sum())

            # Clip long side
            if sector_long > max_weight:
                scale = max_weight / sector_long
                adjusted[sector_mask & (signal > 0)] *= scale

            # Clip short side
            if sector_short > max_weight:
                scale = max_weight / sector_short
                adjusted[sector_mask & (signal < 0)] *= scale

        # Re-normalize to sum to ~0
        long_sum = adjusted[adjusted > 0].sum()
        short_sum = abs(adjusted[adjusted < 0].sum())
        if long_sum > 0 and short_sum > 0:
            target = (long_sum + short_sum) / 2
            adjusted[adjusted > 0] *= target / long_sum
            adjusted[adjusted < 0] *= target / short_sum

        return adjusted

    def check_risk(self, targets: pd.Series, risk_context: Dict[str, Any]) -> RiskCheckResult:
        """Enhanced risk check with leverage and beta constraints."""
        result = super().check_risk(targets, risk_context)

        # Gross leverage check
        max_gross = self._params.get("max_gross_leverage", 2.0) * self._params.get("book_size", 1_000_000)
        gross = targets.abs().sum()
        if gross > max_gross:
            scale = max_gross / gross
            result.clipped_targets = targets * scale
            result.soft_warnings.append(f"Gross leverage {gross/self._params['book_size']:.2f}x exceeds limit")

        # Portfolio beta check
        betas = risk_context.get("betas")
        if betas is not None:
            portfolio_beta = (targets * betas.reindex(targets.index, fill_value=1)).sum() / targets.abs().sum()
            target_beta = self._params.get("target_beta", 0.0)
            if abs(portfolio_beta - target_beta) > 0.15:
                result.soft_warnings.append(
                    f"Portfolio beta {portfolio_beta:.3f} deviates from target {target_beta}"
                )

        return result
