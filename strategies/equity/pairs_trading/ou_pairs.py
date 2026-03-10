"""
EQ_PAIRS_006 — Ornstein-Uhlenbeck Pairs Trading Strategy

Statistical arbitrage strategy that identifies cointegrated stock pairs
and trades the mean-reverting spread using the Ornstein-Uhlenbeck process.

Signal Construction:
    1. Screen universe for candidate pairs (same sector, high correlation)
    2. Test cointegration using Engle-Granger two-step method (ADF on residuals)
    3. Estimate OU parameters: θ (mean reversion speed), μ (long-run mean), σ
    4. Compute z-score of current spread: z = (spread - μ) / σ
    5. Entry: |z| > 2.0 (open position when spread is 2σ from mean)
    6. Exit: |z| < 0.5 (close when spread reverts near mean)
    7. Stop-loss: |z| > 4.0 (close if spread diverges further)

Position Sizing:
    - Equal dollar long/short legs
    - Hedge ratio from cointegration regression
    - Max 15 active pairs at any time

Risk Controls:
    - Cointegration must be significant (p < 0.05)
    - Half-life of mean reversion < 30 days
    - Min correlation > 0.70 over lookback
    - Max holding period: 30 days (force close)

Academic basis:
    - Gatev, Goetzmann & Rouwenhorst (2006): Pairs Trading
    - Vidyamurthy (2004): Pairs Trading — Quantitative Methods
    - Avellaneda & Lee (2010): Statistical Arbitrage in the US Equities Market
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from datetime import date
from itertools import combinations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from base import (
    BaseStrategy, StrategyMetadata, StrategyContext,
    AssetClass, StrategyStyle, RiskCheckResult
)


class OUPairsTrading(BaseStrategy):
    """
    Ornstein-Uhlenbeck Pairs Trading.

    Identifies cointegrated stock pairs and trades the mean-reverting
    spread using OU process parameter estimation.
    """

    DEFAULT_PARAMS = {
        # Pair selection
        "lookback_days": 252,        # Lookback for cointegration test
        "min_correlation": 0.70,     # Minimum correlation for candidate pairs
        "max_pairs": 15,             # Maximum active pairs
        "same_sector_only": True,    # Only pair within same sector

        # Cointegration
        "coint_pvalue": 0.05,        # Max p-value for cointegration
        "max_half_life": 30,         # Max half-life in days

        # OU process signals
        "entry_z": 2.0,              # Entry threshold (z-score)
        "exit_z": 0.5,               # Exit threshold
        "stop_z": 4.0,               # Stop-loss threshold
        "max_holding_days": 30,      # Force close after N days

        # Position sizing
        "book_size": 1_000_000,
        "per_pair_allocation": 0.10, # 10% of book per pair
        "max_single_weight": 0.05,

        # Filters
        "min_price": 10.0,
        "min_volume_avg": 2_000_000,
        "rebalance_frequency": "daily",
    }

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        merged = {**self.DEFAULT_PARAMS, **(params or {})}
        super().__init__(merged)
        self._active_pairs: Dict[str, Dict] = {}  # Track active pair trades

    def get_metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            code="EQ_PAIRS_006",
            name="Ornstein-Uhlenbeck Pairs Trading",
            source_book="151TS",
            asset_class=AssetClass.EQUITY,
            style=StrategyStyle.STATISTICAL_ARB,
            sub_style="cointegration-based pairs trading",
            horizon="daily",
            directionality="long_short",
            complexity="complex",
            description=(
                "Statistical arbitrage using cointegrated stock pairs. "
                "Estimates OU process parameters to trade mean-reverting spreads. "
                "Entry at 2σ, exit at 0.5σ, stop at 4σ. Max 15 active pairs."
            ),
            math_formula=(
                "Spread = P_A - β*P_B; "
                "dS = θ(μ - S)dt + σdW (Ornstein-Uhlenbeck); "
                "z = (S - μ)/σ; "
                "half_life = ln(2)/θ; "
                "Entry: |z| > 2; Exit: |z| < 0.5; Stop: |z| > 4"
            ),
            assumptions=[
                "Cointegration relationship is stable over the trading horizon",
                "Spread follows an OU process with constant parameters",
                "Mean reversion occurs within the half-life estimate",
                "Same-sector pairs have more stable cointegration",
                "Transaction costs don't erode the spread profit",
            ],
            known_failure_modes=[
                "Cointegration breakdown: structural changes break the relationship",
                "Regime change: pairs can diverge permanently after M&A, sector rotation",
                "Crowding: popular pairs have compressed spreads",
                "Execution risk: simultaneous entry/exit of both legs",
                "Short squeeze risk on the short leg",
            ],
            capacity_notes="Moderate capacity; limited by pair availability and spread width",
            required_data=["bars_1d", "sector_classification"],
            parameters=self._params,
            parameter_bounds={
                "lookback_days": (126, 504),
                "min_correlation": (0.50, 0.90),
                "max_pairs": (5, 30),
                "coint_pvalue": (0.01, 0.10),
                "max_half_life": (10, 60),
                "entry_z": (1.5, 3.0),
                "exit_z": (0.0, 1.0),
                "stop_z": (3.0, 5.0),
            },
        )

    def generate_features(self, context: StrategyContext, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Identify cointegrated pairs and compute spread features.

        Features generated (per pair):
            - pair_id: "TICKER_A|TICKER_B"
            - correlation: Rolling correlation
            - hedge_ratio: Cointegration regression beta
            - spread: Current spread value
            - spread_mean: Long-run mean of spread
            - spread_std: Standard deviation of spread
            - z_score: Current z-score of spread
            - half_life: Estimated half-life of mean reversion
            - coint_pvalue: Cointegration test p-value
            - ou_theta: OU mean reversion speed
        """
        prices = data.get("close")
        if prices is None or prices.empty:
            raise ValueError("Price data ('close') is required")

        lookback = self._params["lookback_days"]
        min_corr = self._params["min_correlation"]
        coint_pval = self._params["coint_pvalue"]
        max_hl = self._params["max_half_life"]
        same_sector = self._params["same_sector_only"]

        # Use recent data
        recent = prices.iloc[-lookback:] if len(prices) > lookback else prices

        # Filter valid tickers
        valid = recent.columns[recent.isna().sum() < len(recent) * 0.1]
        recent = recent[valid]

        # Get sectors
        sectors = data.get("sectors")

        # Step 1: Find candidate pairs by correlation
        candidates = self._find_candidate_pairs(recent, sectors, min_corr, same_sector)

        if not candidates:
            return pd.DataFrame()

        # Step 2: Test cointegration and compute OU parameters
        pair_features = []
        for ticker_a, ticker_b, corr in candidates:
            pa = recent[ticker_a].dropna()
            pb = recent[ticker_b].dropna()
            common = pa.index.intersection(pb.index)
            if len(common) < lookback * 0.5:
                continue

            pa = pa.loc[common]
            pb = pb.loc[common]

            # Cointegration test (Engle-Granger)
            coint_result = self._engle_granger_test(pa, pb)
            if coint_result is None:
                continue

            pvalue, hedge_ratio, residuals = coint_result

            if pvalue > coint_pval:
                continue

            # OU parameter estimation
            ou_params = self._estimate_ou_params(residuals)
            if ou_params is None:
                continue

            theta, mu, sigma = ou_params
            half_life = np.log(2) / theta if theta > 0 else float("inf")

            if half_life > max_hl or half_life < 1:
                continue

            # Current spread and z-score
            spread = pa.iloc[-1] - hedge_ratio * pb.iloc[-1]
            z_score = (spread - mu) / sigma if sigma > 0 else 0

            pair_features.append({
                "pair_id": f"{ticker_a}|{ticker_b}",
                "ticker_a": ticker_a,
                "ticker_b": ticker_b,
                "correlation": corr,
                "hedge_ratio": hedge_ratio,
                "spread": spread,
                "spread_mean": mu,
                "spread_std": sigma,
                "z_score": z_score,
                "half_life": half_life,
                "coint_pvalue": pvalue,
                "ou_theta": theta,
                "price_a": pa.iloc[-1],
                "price_b": pb.iloc[-1],
            })

        if not pair_features:
            return pd.DataFrame()

        features = pd.DataFrame(pair_features)
        features.index = features["pair_id"]

        # Sort by cointegration strength (lower p-value = better)
        features = features.sort_values("coint_pvalue")

        return features

    def _find_candidate_pairs(
        self,
        prices: pd.DataFrame,
        sectors: Optional[pd.Series],
        min_corr: float,
        same_sector: bool,
    ) -> List[Tuple[str, str, float]]:
        """Find candidate pairs by correlation and sector."""
        tickers = list(prices.columns)
        n = len(tickers)

        if n > 100:
            # For large universes, use correlation matrix approach
            corr_matrix = prices.pct_change().dropna().corr()
        else:
            corr_matrix = prices.pct_change().dropna().corr()

        candidates = []
        for i in range(n):
            for j in range(i + 1, n):
                ta, tb = tickers[i], tickers[j]
                corr = corr_matrix.iloc[i, j]

                if abs(corr) < min_corr:
                    continue

                # Sector filter
                if same_sector and sectors is not None:
                    sa = sectors.get(ta)
                    sb = sectors.get(tb)
                    if sa is None or sb is None or sa != sb:
                        continue

                candidates.append((ta, tb, corr))

        # Sort by correlation (descending) and limit
        candidates.sort(key=lambda x: -abs(x[2]))
        return candidates[:200]  # Top 200 candidates for testing

    def _engle_granger_test(
        self,
        y: pd.Series,
        x: pd.Series,
    ) -> Optional[Tuple[float, float, pd.Series]]:
        """
        Engle-Granger two-step cointegration test.
        Returns (p_value, hedge_ratio, residuals) or None.
        """
        try:
            # Step 1: OLS regression y = α + β*x + ε
            X = np.column_stack([np.ones(len(x)), x.values])
            beta, _, _, _ = np.linalg.lstsq(X, y.values, rcond=None)
            hedge_ratio = beta[1]
            residuals = y.values - X @ beta

            # Step 2: ADF test on residuals (simplified)
            # Using Dickey-Fuller regression: Δε_t = γ*ε_{t-1} + u_t
            resid_series = pd.Series(residuals, index=y.index)
            delta_resid = resid_series.diff().dropna()
            lagged_resid = resid_series.shift(1).dropna()

            # Align
            common = delta_resid.index.intersection(lagged_resid.index)
            dr = delta_resid.loc[common].values
            lr = lagged_resid.loc[common].values

            # OLS: Δε = γ*ε_{t-1}
            X_adf = lr.reshape(-1, 1)
            gamma_hat = np.linalg.lstsq(X_adf, dr, rcond=None)[0][0]

            # t-statistic
            fitted = X_adf @ np.array([gamma_hat])
            residuals_adf = dr - fitted.flatten()
            se = np.sqrt(np.sum(residuals_adf ** 2) / (len(dr) - 1)) / np.sqrt(np.sum(lr ** 2))
            t_stat = gamma_hat / se if se > 0 else 0

            # Approximate p-value using MacKinnon critical values
            # For n=2 (two variables), approximate critical values:
            # 1%: -3.90, 5%: -3.34, 10%: -3.05
            if t_stat < -3.90:
                p_value = 0.01
            elif t_stat < -3.34:
                p_value = 0.05
            elif t_stat < -3.05:
                p_value = 0.10
            else:
                p_value = 0.50  # Not cointegrated

            return (p_value, hedge_ratio, pd.Series(residuals, index=y.index))

        except (np.linalg.LinAlgError, ValueError):
            return None

    def _estimate_ou_params(
        self,
        residuals: pd.Series,
    ) -> Optional[Tuple[float, float, float]]:
        """
        Estimate Ornstein-Uhlenbeck parameters from residuals.
        dS = θ(μ - S)dt + σdW

        Returns (theta, mu, sigma) or None.
        """
        try:
            s = residuals.values
            n = len(s)
            if n < 30:
                return None

            # MLE estimation
            # AR(1) regression: S_t = a + b*S_{t-1} + ε
            s_lag = s[:-1]
            s_curr = s[1:]

            X = np.column_stack([np.ones(len(s_lag)), s_lag])
            beta, _, _, _ = np.linalg.lstsq(X, s_curr, rcond=None)
            a, b = beta

            # OU parameters
            dt = 1.0 / 252  # Daily
            if b <= 0 or b >= 1:
                return None

            theta = -np.log(b) / dt
            mu = a / (1 - b)
            residuals_ar = s_curr - X @ beta
            sigma_e = np.std(residuals_ar)
            sigma = sigma_e * np.sqrt(2 * theta / (1 - b ** 2)) if (1 - b ** 2) > 0 else sigma_e

            if theta <= 0 or sigma <= 0:
                return None

            return (theta, mu, sigma)

        except (np.linalg.LinAlgError, ValueError):
            return None

    def generate_signal(self, features: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Generate pairs trading signal based on spread z-scores.

        For each qualifying pair:
            - If z > entry_z: short spread (short A, long B)
            - If z < -entry_z: long spread (long A, short B)
            - Signal strength proportional to |z|

        Returns a signal indexed by individual tickers (not pairs).
        """
        entry_z = params.get("entry_z", 2.0)
        stop_z = params.get("stop_z", 4.0)
        max_pairs = params.get("max_pairs", 15)

        if len(features) == 0:
            return pd.Series(dtype=float)

        # Collect all unique tickers
        all_tickers = set()
        for _, row in features.iterrows():
            all_tickers.add(row["ticker_a"])
            all_tickers.add(row["ticker_b"])

        signal = pd.Series(0.0, index=sorted(all_tickers))

        # Select top pairs by cointegration quality
        active_pairs = features.head(max_pairs)
        n_active = 0

        for _, pair in active_pairs.iterrows():
            z = pair["z_score"]
            ta = pair["ticker_a"]
            tb = pair["ticker_b"]
            hr = pair["hedge_ratio"]

            # Skip if beyond stop-loss
            if abs(z) > stop_z:
                continue

            # Entry signals
            if z > entry_z:
                # Spread too high → short A, long B (expect convergence)
                weight = min(abs(z) / entry_z - 1, 1.0)  # 0 to 1
                signal[ta] -= weight / max_pairs
                signal[tb] += weight * hr / max_pairs
                n_active += 1
            elif z < -entry_z:
                # Spread too low → long A, short B
                weight = min(abs(z) / entry_z - 1, 1.0)
                signal[ta] += weight / max_pairs
                signal[tb] -= weight * hr / max_pairs
                n_active += 1

        return signal

    def check_risk(self, targets: pd.Series, risk_context: Dict[str, Any]) -> RiskCheckResult:
        """Risk check with pair-specific constraints."""
        result = super().check_risk(targets, risk_context)

        # Check that we're approximately market-neutral
        net = targets.sum()
        gross = targets.abs().sum()
        if gross > 0 and abs(net) / gross > 0.15:
            result.soft_warnings.append(
                f"Net/Gross ratio {abs(net)/gross:.2%} exceeds 15% threshold"
            )

        return result
