"""
EQ_RMOM_004 — Residual Momentum Strategy

Ranks stocks by momentum in their Fama-French 3-factor residuals rather
than raw returns. This isolates stock-specific momentum from factor
momentum, producing a signal that is orthogonal to market, size, and
value factors.

Signal Construction:
    1. Run rolling 252-day Fama-French 3-factor regression for each stock:
       r_i - r_f = α_i + β_mkt*(r_mkt - r_f) + β_smb*SMB + β_hml*HML + ε_i
    2. Compute cumulative residual return over months t-12 to t-1:
       residual_mom = Σ(ε_i) for t-252 to t-21
    3. Cross-sectional z-score and rank
    4. Long top decile, short bottom decile

Academic basis:
    - Blitz, Huij & Martens (2011): Residual Momentum
    - Gutierrez & Pirinsky (2007): Momentum and idiosyncratic risk
    - Key insight: residual momentum captures stock-specific information flow,
      not factor rotation, making it more robust to momentum crashes
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


class ResidualMomentum(BaseStrategy):
    """
    Residual Momentum (Fama-French 3-Factor Residuals).

    Isolates stock-specific momentum by regressing out market, size,
    and value factor exposures. More robust to momentum crashes than
    raw cross-sectional momentum.
    """

    DEFAULT_PARAMS = {
        "regression_window": 252,    # Rolling regression window
        "signal_window": 252,        # Residual accumulation window
        "skip_days": 21,             # Skip most recent month
        "min_observations": 180,     # Minimum obs for valid regression
        "top_pct": 0.10,             # Long top 10%
        "bottom_pct": 0.10,          # Short bottom 10%
        "book_size": 1_000_000,
        "max_single_weight": 0.05,
        "sector_neutral": True,
        "min_price": 5.0,
        "min_volume_avg": 1_000_000,
        "min_r_squared": 0.10,       # Minimum R² for valid regression
        "weighting": "equal",
        "rebalance_frequency": "monthly",
    }

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        merged = {**self.DEFAULT_PARAMS, **(params or {})}
        super().__init__(merged)

    def get_metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            code="EQ_RMOM_004",
            name="Residual Momentum",
            source_book="151TS",
            asset_class=AssetClass.EQUITY,
            style=StrategyStyle.MOMENTUM,
            sub_style="Fama-French residual momentum",
            horizon="monthly",
            directionality="long_short",
            complexity="complex",
            description=(
                "Ranks stocks by momentum in Fama-French 3-factor residuals. "
                "Isolates stock-specific momentum from factor momentum. "
                "More crash-resistant than raw momentum. Monthly rebalance."
            ),
            math_formula=(
                "r_i - r_f = α + β_mkt*(r_mkt-r_f) + β_smb*SMB + β_hml*HML + ε; "
                "residual_mom = Σε(t-252 to t-21); signal = z(residual_mom)"
            ),
            assumptions=[
                "Stock-specific momentum (residual) is distinct from factor momentum",
                "Fama-French 3-factor model captures systematic risk adequately",
                "Residual momentum is more robust to momentum crashes",
                "Information diffusion drives stock-specific momentum",
            ],
            known_failure_modes=[
                "Requires reliable Fama-French factor data (Kenneth French library)",
                "Low R² regressions produce noisy residuals",
                "Higher turnover than raw momentum due to residual noise",
                "Underperforms when factor momentum dominates stock-specific momentum",
            ],
            capacity_notes="Moderate capacity; higher turnover reduces effective capacity",
            required_data=["bars_1d", "ff_factors", "sector_classification"],
            parameters=self._params,
            parameter_bounds={
                "regression_window": (126, 504),
                "signal_window": (126, 504),
                "skip_days": (0, 63),
                "min_observations": (63, 252),
                "top_pct": (0.05, 0.30),
                "bottom_pct": (0.05, 0.30),
                "min_r_squared": (0.0, 0.50),
            },
        )

    def generate_features(self, context: StrategyContext, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Compute residual momentum features using Fama-French regression.

        Features generated:
            - residual_mom: Cumulative residual return (12-1 month)
            - residual_mom_z: Cross-sectional z-score
            - alpha: Regression intercept (annualized)
            - beta_mkt: Market beta
            - beta_smb: Size beta
            - beta_hml: Value beta
            - r_squared: Regression R²
            - idio_vol: Idiosyncratic volatility (annualized)
        """
        prices = data.get("close")
        ff_factors = data.get("ff_factors")

        if prices is None or prices.empty:
            raise ValueError("Price data ('close') is required")

        # Daily returns
        returns = prices.pct_change().dropna(how="all")

        # Fama-French factors
        if ff_factors is not None and not ff_factors.empty:
            mkt_rf = ff_factors.get("Mkt-RF", ff_factors.get("mkt_rf", pd.Series(dtype=float)))
            smb = ff_factors.get("SMB", ff_factors.get("smb", pd.Series(dtype=float)))
            hml = ff_factors.get("HML", ff_factors.get("hml", pd.Series(dtype=float)))
            rf = ff_factors.get("RF", ff_factors.get("rf", pd.Series(0, index=returns.index)))
        else:
            # Fallback: construct approximate factors from data
            mkt_rf, smb, hml, rf = self._construct_approximate_factors(returns, data)

        reg_window = self._params["regression_window"]
        sig_window = self._params["signal_window"]
        skip = self._params["skip_days"]
        min_obs = self._params["min_observations"]
        min_r2 = self._params["min_r_squared"]

        # Align all data
        common_idx = returns.index
        for s in [mkt_rf, smb, hml, rf]:
            if s is not None and len(s) > 0:
                common_idx = common_idx.intersection(s.index)

        returns_aligned = returns.loc[common_idx]
        mkt_rf_aligned = mkt_rf.reindex(common_idx, fill_value=0)
        smb_aligned = smb.reindex(common_idx, fill_value=0)
        hml_aligned = hml.reindex(common_idx, fill_value=0)
        rf_aligned = rf.reindex(common_idx, fill_value=0)

        # Use the signal window for residual computation
        start_idx = max(0, len(returns_aligned) - sig_window - skip)
        end_idx = len(returns_aligned) - skip if skip > 0 else len(returns_aligned)

        ret_window = returns_aligned.iloc[start_idx:end_idx]
        mkt_window = mkt_rf_aligned.iloc[start_idx:end_idx]
        smb_window = smb_aligned.iloc[start_idx:end_idx]
        hml_window = hml_aligned.iloc[start_idx:end_idx]
        rf_window = rf_aligned.iloc[start_idx:end_idx]

        # Run regressions for each stock
        results = {}
        for ticker in returns.columns:
            stock_ret = ret_window[ticker].dropna()
            if len(stock_ret) < min_obs:
                continue

            # Excess returns
            excess_ret = stock_ret - rf_window.reindex(stock_ret.index, fill_value=0)

            # Build factor matrix
            X = pd.DataFrame({
                "mkt_rf": mkt_window.reindex(stock_ret.index, fill_value=0),
                "smb": smb_window.reindex(stock_ret.index, fill_value=0),
                "hml": hml_window.reindex(stock_ret.index, fill_value=0),
            })

            # Drop rows with NaN
            valid = excess_ret.dropna().index.intersection(X.dropna().index)
            if len(valid) < min_obs:
                continue

            y = excess_ret.loc[valid].values
            X_mat = X.loc[valid].values
            X_with_const = np.column_stack([np.ones(len(valid)), X_mat])

            # OLS regression
            try:
                beta, residuals_arr, rank, sv = np.linalg.lstsq(X_with_const, y, rcond=None)

                fitted = X_with_const @ beta
                resid = y - fitted

                # R-squared
                ss_res = np.sum(resid ** 2)
                ss_tot = np.sum((y - y.mean()) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

                if r_squared < min_r2:
                    continue

                # Cumulative residual return
                residual_mom = np.sum(resid)

                # Idiosyncratic volatility
                idio_vol = np.std(resid) * np.sqrt(252)

                results[ticker] = {
                    "residual_mom": residual_mom,
                    "alpha": beta[0] * 252,  # Annualized
                    "beta_mkt": beta[1],
                    "beta_smb": beta[2],
                    "beta_hml": beta[3],
                    "r_squared": r_squared,
                    "idio_vol": idio_vol,
                }
            except (np.linalg.LinAlgError, ValueError):
                continue

        if not results:
            return pd.DataFrame(columns=[
                "residual_mom", "residual_mom_z", "alpha",
                "beta_mkt", "beta_smb", "beta_hml", "r_squared", "idio_vol"
            ])

        features = pd.DataFrame(results).T

        # Cross-sectional z-score of residual momentum
        rm = features["residual_mom"]
        rm_clean = rm.replace([np.inf, -np.inf], np.nan).dropna()
        if len(rm_clean) > 5 and rm_clean.std() > 0:
            lower = rm_clean.quantile(0.01)
            upper = rm_clean.quantile(0.99)
            winsorized = rm_clean.clip(lower, upper)
            features["residual_mom_z"] = ((winsorized - winsorized.mean()) / winsorized.std()).clip(-3, 3)
        else:
            features["residual_mom_z"] = 0

        # Add price and volume for filtering
        if prices is not None:
            features["last_price"] = prices.iloc[-1].reindex(features.index)
        volumes = data.get("volume")
        if volumes is not None:
            features["avg_volume_20d"] = volumes.iloc[-20:].mean().reindex(features.index)
        sectors = data.get("sectors")
        if sectors is not None:
            features["sector"] = sectors.reindex(features.index)

        return features

    def _construct_approximate_factors(self, returns: pd.DataFrame, data: Dict[str, pd.DataFrame]):
        """
        Construct approximate Fama-French factors from available data.
        Used as fallback when Kenneth French factor data is unavailable.
        """
        # Market factor: equal-weighted market return
        mkt_rf = returns.mean(axis=1)

        # SMB: approximate using market cap data if available
        smb = pd.Series(0, index=returns.index)
        hml = pd.Series(0, index=returns.index)
        rf = pd.Series(0, index=returns.index)

        # If we have market cap data, construct better approximations
        market_caps = data.get("market_cap")
        if market_caps is not None and not market_caps.empty:
            median_cap = market_caps.median()
            small = returns.columns[market_caps < median_cap] if isinstance(market_caps, pd.Series) else []
            big = returns.columns[market_caps >= median_cap] if isinstance(market_caps, pd.Series) else []

            if len(small) > 0 and len(big) > 0:
                smb = returns[small].mean(axis=1) - returns[big].mean(axis=1)

        return mkt_rf, smb, hml, rf

    def generate_signal(self, features: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Generate residual momentum signal.

        Process:
            1. Filter by price, volume, and regression quality
            2. Rank by residual_mom_z
            3. Long top decile, short bottom decile
            4. Apply sector-neutral adjustment
        """
        min_price = params.get("min_price", 5.0)
        min_volume = params.get("min_volume_avg", 1_000_000)
        top_pct = params.get("top_pct", 0.10)
        bottom_pct = params.get("bottom_pct", 0.10)
        sector_neutral = params.get("sector_neutral", True)

        # Filters
        mask = pd.Series(True, index=features.index)
        if "last_price" in features.columns:
            mask &= features["last_price"] >= min_price
        if "avg_volume_20d" in features.columns:
            mask &= features["avg_volume_20d"] >= min_volume

        filtered = features[mask].dropna(subset=["residual_mom_z"])

        if len(filtered) < 20:
            return pd.Series(0, index=features.index, dtype=float)

        # Rank by residual momentum z-score
        z_scores = filtered["residual_mom_z"]
        n = len(z_scores)
        n_long = max(int(n * top_pct), 1)
        n_short = max(int(n * bottom_pct), 1)

        ranked = z_scores.sort_values(ascending=False)
        long_names = ranked.head(n_long).index
        short_names = ranked.tail(n_short).index

        # Build signal
        signal = pd.Series(0.0, index=features.index)

        weighting = params.get("weighting", "equal")
        if weighting == "equal":
            signal[long_names] = 1.0 / n_long
            signal[short_names] = -1.0 / n_short
        else:
            long_z = z_scores[long_names]
            short_z = z_scores[short_names]
            if long_z.abs().sum() > 0:
                signal[long_names] = long_z / long_z.abs().sum()
            if short_z.abs().sum() > 0:
                signal[short_names] = short_z / short_z.abs().sum()

        # Sector-neutral adjustment
        if sector_neutral and "sector" in filtered.columns:
            signal = self._sector_neutralize(signal, features.get("sector", pd.Series()))

        return signal

    def _sector_neutralize(self, signal: pd.Series, sectors: pd.Series) -> pd.Series:
        """Demean signal within each sector for neutrality."""
        adjusted = signal.copy()
        for sector in sectors.dropna().unique():
            sector_mask = sectors == sector
            sector_signal = signal[sector_mask]
            if len(sector_signal) > 1:
                adjusted[sector_mask] = sector_signal - sector_signal.mean()

        long_sum = adjusted[adjusted > 0].sum()
        short_sum = abs(adjusted[adjusted < 0].sum())
        if long_sum > 0 and short_sum > 0:
            adjusted[adjusted > 0] /= long_sum
            adjusted[adjusted < 0] /= short_sum

        return adjusted
