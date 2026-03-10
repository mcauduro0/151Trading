"""Dataset Manifest Schema.

Defines the format for dataset manifests used by the backtest engine
to ensure reproducible, point-in-time data access.

Each manifest specifies:
- What data sources to use
- How to handle adjustments (splits, dividends)
- Point-in-time as-of timestamp
- Universe definition
"""

from datetime import date, datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DataSourceSpec(BaseModel):
    """Specification for a single data source within a manifest."""
    provider: str  # yahoo_finance, fred, fmp, polygon, etc.
    dataset: str  # bars_1d, fundamentals, macro_series, etc.
    symbols: Optional[List[str]] = None  # None means all in universe
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    fields: Optional[List[str]] = None  # None means all fields
    filters: Dict[str, Any] = Field(default_factory=dict)


class AdjustmentPolicy(BaseModel):
    """How to handle corporate actions and data adjustments."""
    split_adjust: bool = True
    dividend_adjust: bool = True
    currency_convert: bool = False
    target_currency: str = "USD"
    fill_method: str = "ffill"  # ffill, bfill, interpolate, none
    max_fill_days: int = 5


class UniverseSpec(BaseModel):
    """Universe definition for the dataset."""
    universe_id: Optional[str] = None
    asset_class: Optional[str] = None
    symbols: Optional[List[str]] = None
    filters: Dict[str, Any] = Field(default_factory=dict)
    min_history_days: int = 252  # Minimum history required
    min_avg_volume: Optional[float] = None
    min_market_cap: Optional[float] = None


class DatasetManifest(BaseModel):
    """Complete dataset manifest for reproducible backtesting.

    Example:
        manifest = DatasetManifest(
            description="US Large Cap Momentum - 2020-2025",
            sources=[
                DataSourceSpec(provider="yahoo_finance", dataset="bars_1d"),
                DataSourceSpec(provider="fmp", dataset="fundamentals"),
                DataSourceSpec(provider="fred", dataset="macro_series",
                              symbols=["DGS10", "VIXCLS", "T10Y2Y"]),
            ],
            universe=UniverseSpec(
                asset_class="equity",
                min_market_cap=10_000_000_000,
                min_avg_volume=1_000_000,
            ),
            adjustment_policy=AdjustmentPolicy(split_adjust=True, dividend_adjust=True),
            as_of=datetime(2025, 12, 31),
        )
    """
    description: str
    sources: List[DataSourceSpec]
    universe: Optional[UniverseSpec] = None
    adjustment_policy: AdjustmentPolicy = Field(default_factory=AdjustmentPolicy)
    as_of: Optional[datetime] = None  # Point-in-time timestamp
    benchmark_symbols: List[str] = Field(default_factory=lambda: ["SPY"])
    created_at: datetime = Field(default_factory=datetime.utcnow)
    version: int = 1
    tags: List[str] = Field(default_factory=list)

    class Config:
        json_schema_extra = {
            "example": {
                "description": "US Large Cap Momentum Backtest Data",
                "sources": [
                    {"provider": "yahoo_finance", "dataset": "bars_1d"},
                    {"provider": "fmp", "dataset": "fundamentals"},
                ],
                "universe": {
                    "asset_class": "equity",
                    "min_market_cap": 10000000000,
                },
                "adjustment_policy": {
                    "split_adjust": True,
                    "dividend_adjust": True,
                },
                "benchmark_symbols": ["SPY", "QQQ"],
            }
        }
