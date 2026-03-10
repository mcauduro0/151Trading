"""
EQ_VAL_002 — Enhanced Value Composite Strategy

Multi-factor value strategy combining four classic value metrics:
    1. Book-to-Market (B/M)
    2. Earnings Yield (E/P)
    3. Cash Flow Yield (CF/P)
    4. Sales-to-Price (S/P)

Applies a quality filter (Piotroski F-Score >= 5) to avoid value traps.
Rebalances quarterly with sector-neutral constraints.

Signal Construction:
    For each metric:
        z_i = (metric_i - mean) / std   [cross-sectional z-score]
    composite = 0.30 * z_BM + 0.25 * z_EP + 0.25 * z_CFP + 0.20 * z_SP
    quality_filter: F-Score >= 5 (avoid value traps)
    signal = composite * quality_mask

Academic basis:
    - Fama & French (1993): Book-to-market factor
    - Lakonishok, Shleifer & Vishny (1994): Contrarian investment
    - Piotroski (2000): F-Score quality filter
    - Asness, Moskowitz & Pedersen (2013): Value and Momentum Everywhere
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


class EnhancedValueComposite(BaseStrategy):
    """
    Enhanced Value Composite with Quality Filter.

    Combines four value metrics into a composite score and applies
    Piotroski F-Score quality filter to avoid value traps.
    """

    DEFAULT_PARAMS = {
        "weight_bm": 0.30,          # Book-to-Market weight
        "weight_ep": 0.25,          # Earnings Yield weight
        "weight_cfp": 0.25,         # Cash Flow Yield weight
        "weight_sp": 0.20,          # Sales-to-Price weight
        "min_fscore": 5,            # Minimum Piotroski F-Score
        "top_pct": 0.10,            # Long top 10%
        "bottom_pct": 0.10,         # Short bottom 10%
        "book_size": 1_000_000,
        "max_single_weight": 0.05,
        "sector_neutral": True,
        "min_market_cap": 2e9,      # $2B minimum market cap
        "exclude_financials": False, # Optionally exclude financials (B/M distortion)
        "rebalance_frequency": "quarterly",
    }

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        merged = {**self.DEFAULT_PARAMS, **(params or {})}
        super().__init__(merged)

    def get_metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            code="EQ_VAL_002",
            name="Enhanced Value Composite",
            source_book="151TS",
            asset_class=AssetClass.EQUITY,
            style=StrategyStyle.VALUE,
            sub_style="multi-factor composite with quality filter",
            horizon="quarterly",
            directionality="long_short",
            complexity="moderate",
            description=(
                "Multi-factor value strategy combining book-to-market, earnings yield, "
                "cash flow yield, and sales-to-price. Applies Piotroski F-Score quality "
                "filter to avoid value traps. Quarterly rebalance, sector-neutral."
            ),
            math_formula=(
                "V = 0.30*z(B/M) + 0.25*z(E/P) + 0.25*z(CF/P) + 0.20*z(S/P); "
                "quality_gate: F-Score >= 5"
            ),
            assumptions=[
                "Value premium persists in US equities over medium horizons",
                "Composite score is more robust than single-metric value",
                "Piotroski F-Score effectively filters value traps",
                "Quarterly rebalance captures value convergence without excess turnover",
            ],
            known_failure_modes=[
                "Value drawdowns can be prolonged (2017-2020 growth dominance)",
                "Book value distorted for financials and asset-light tech companies",
                "Earnings manipulation can corrupt E/P signal",
                "Sector concentration risk in deep value regimes",
            ],
            capacity_notes="High capacity in large-cap, lower in small-cap due to liquidity",
            required_data=["fundamentals", "bars_1d", "sector_classification"],
            parameters=self._params,
            parameter_bounds={
                "weight_bm": (0.0, 0.50),
                "weight_ep": (0.0, 0.50),
                "weight_cfp": (0.0, 0.50),
                "weight_sp": (0.0, 0.50),
                "min_fscore": (0, 9),
                "top_pct": (0.05, 0.30),
                "bottom_pct": (0.05, 0.30),
            },
        )

    def generate_features(self, context: StrategyContext, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Compute value features from fundamental data.

        Features generated:
            - book_to_market: Book value / Market cap
            - earnings_yield: Net income / Market cap (E/P)
            - cf_yield: Operating cash flow / Market cap
            - sales_to_price: Revenue / Market cap
            - f_score: Piotroski F-Score (0-9)
            - market_cap: Market capitalization
            - sector: GICS sector
        """
        fundamentals = data.get("fundamentals")
        prices = data.get("close")

        if fundamentals is None:
            raise ValueError("Fundamental data is required for value strategy")

        features = pd.DataFrame(index=fundamentals.index)

        # Extract fundamental metrics
        # These columns should be present from FMP or Yahoo data
        features["book_to_market"] = fundamentals.get("bookValuePerShare", 0) / fundamentals.get("price", 1).replace(0, np.nan)
        features["earnings_yield"] = fundamentals.get("netIncomePerShare", 0) / fundamentals.get("price", 1).replace(0, np.nan)
        features["cf_yield"] = fundamentals.get("operatingCashFlowPerShare", 0) / fundamentals.get("price", 1).replace(0, np.nan)
        features["sales_to_price"] = fundamentals.get("revenuePerShare", 0) / fundamentals.get("price", 1).replace(0, np.nan)
        features["market_cap"] = fundamentals.get("marketCap", 0)

        # Compute Piotroski F-Score
        features["f_score"] = self._compute_fscore(fundamentals)

        # Sector
        if "sector" in fundamentals.columns:
            features["sector"] = fundamentals["sector"]

        # Cross-sectional z-scores for each value metric
        for col in ["book_to_market", "earnings_yield", "cf_yield", "sales_to_price"]:
            valid = features[col].replace([np.inf, -np.inf], np.nan).dropna()
            if len(valid) > 5:
                # Winsorize at 1st/99th percentile before z-scoring
                lower = valid.quantile(0.01)
                upper = valid.quantile(0.99)
                winsorized = valid.clip(lower, upper)
                z = (winsorized - winsorized.mean()) / winsorized.std()
                features[f"{col}_z"] = z
            else:
                features[f"{col}_z"] = 0

        return features

    def _compute_fscore(self, fundamentals: pd.DataFrame) -> pd.Series:
        """
        Compute Piotroski F-Score (0-9) from fundamental data.

        Components:
            Profitability (4 points):
                1. ROA > 0
                2. Operating Cash Flow > 0
                3. ROA improvement YoY
                4. Accruals: CFO > ROA (quality of earnings)
            Leverage/Liquidity (3 points):
                5. Decrease in leverage (LT debt / total assets)
                6. Increase in current ratio
                7. No new equity issuance
            Operating Efficiency (2 points):
                8. Increase in gross margin
                9. Increase in asset turnover
        """
        score = pd.Series(0, index=fundamentals.index, dtype=int)

        # Safely extract with defaults
        def safe_col(name, default=0):
            return fundamentals.get(name, pd.Series(default, index=fundamentals.index))

        roa = safe_col("returnOnAssets", 0)
        roa_prev = safe_col("returnOnAssets_prev", 0)
        cfo = safe_col("operatingCashFlowPerShare", 0)
        net_income = safe_col("netIncomePerShare", 0)
        lt_debt = safe_col("longTermDebt", 0)
        lt_debt_prev = safe_col("longTermDebt_prev", 0)
        total_assets = safe_col("totalAssets", 1)
        total_assets_prev = safe_col("totalAssets_prev", 1)
        current_ratio = safe_col("currentRatio", 1)
        current_ratio_prev = safe_col("currentRatio_prev", 1)
        shares = safe_col("weightedAverageShsOut", 1)
        shares_prev = safe_col("weightedAverageShsOut_prev", 1)
        gross_margin = safe_col("grossProfitMargin", 0)
        gross_margin_prev = safe_col("grossProfitMargin_prev", 0)
        revenue = safe_col("revenue", 0)
        revenue_prev = safe_col("revenue_prev", 0)

        # 1. ROA > 0
        score += (roa > 0).astype(int)
        # 2. CFO > 0
        score += (cfo > 0).astype(int)
        # 3. ROA improvement
        score += (roa > roa_prev).astype(int)
        # 4. Accruals: CFO > Net Income (quality)
        score += (cfo > net_income).astype(int)
        # 5. Leverage decrease
        leverage = lt_debt / total_assets.replace(0, np.nan)
        leverage_prev = lt_debt_prev / total_assets_prev.replace(0, np.nan)
        score += (leverage < leverage_prev).astype(int)
        # 6. Current ratio increase
        score += (current_ratio > current_ratio_prev).astype(int)
        # 7. No dilution
        score += (shares <= shares_prev).astype(int)
        # 8. Gross margin increase
        score += (gross_margin > gross_margin_prev).astype(int)
        # 9. Asset turnover increase
        turnover = revenue / total_assets.replace(0, np.nan)
        turnover_prev = revenue_prev / total_assets_prev.replace(0, np.nan)
        score += (turnover > turnover_prev).astype(int)

        return score.clip(0, 9)

    def generate_signal(self, features: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Generate value composite signal.

        Process:
            1. Filter by market cap and quality (F-Score)
            2. Compute weighted composite of z-scored value metrics
            3. Rank and select top/bottom deciles
            4. Apply sector-neutral adjustment
        """
        min_cap = params.get("min_market_cap", 2e9)
        min_fscore = params.get("min_fscore", 5)
        top_pct = params.get("top_pct", 0.10)
        bottom_pct = params.get("bottom_pct", 0.10)
        sector_neutral = params.get("sector_neutral", True)
        exclude_fin = params.get("exclude_financials", False)

        # Filters
        mask = pd.Series(True, index=features.index)
        if "market_cap" in features.columns:
            mask &= features["market_cap"] >= min_cap
        if "f_score" in features.columns:
            # Quality gate: only long stocks with F-Score >= threshold
            quality_mask = features["f_score"] >= min_fscore
        else:
            quality_mask = pd.Series(True, index=features.index)

        if exclude_fin and "sector" in features.columns:
            mask &= features["sector"] != "Financial Services"

        filtered = features[mask].copy()

        if len(filtered) < 20:
            return pd.Series(0, index=features.index, dtype=float)

        # Weighted composite
        w_bm = params.get("weight_bm", 0.30)
        w_ep = params.get("weight_ep", 0.25)
        w_cfp = params.get("weight_cfp", 0.25)
        w_sp = params.get("weight_sp", 0.20)

        composite = (
            w_bm * filtered.get("book_to_market_z", 0) +
            w_ep * filtered.get("earnings_yield_z", 0) +
            w_cfp * filtered.get("cf_yield_z", 0) +
            w_sp * filtered.get("sales_to_price_z", 0)
        )

        # Rank
        n = len(composite)
        n_long = max(int(n * top_pct), 1)
        n_short = max(int(n * bottom_pct), 1)

        ranked = composite.sort_values(ascending=False)

        # Long: high value + quality gate
        long_candidates = ranked.head(n_long * 2)  # Over-select then filter
        long_names = long_candidates[long_candidates.index.isin(quality_mask[quality_mask].index)].head(n_long).index

        # Short: low value (no quality gate needed for shorts)
        short_names = ranked.tail(n_short).index

        # Build signal
        signal = pd.Series(0.0, index=features.index)
        if len(long_names) > 0:
            signal[long_names] = 1.0 / len(long_names)
        if len(short_names) > 0:
            signal[short_names] = -1.0 / len(short_names)

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

        # Re-normalize
        long_sum = adjusted[adjusted > 0].sum()
        short_sum = adjusted[adjusted < 0].sum()
        if long_sum > 0 and short_sum < 0:
            adjusted[adjusted > 0] /= long_sum
            adjusted[adjusted < 0] /= abs(short_sum)

        return adjusted
