"""Base class for all data provider adapters.

Every data provider adapter must implement this interface to ensure
consistent ingestion, normalization, and quality checking across all sources.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import pandas as pd


@dataclass
class IngestionResult:
    """Result of a data ingestion operation."""
    provider: str
    status: str  # success, partial, failed
    records_processed: int = 0
    records_inserted: int = 0
    records_updated: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def complete(self, status: str = "success"):
        self.status = status
        self.completed_at = datetime.now(timezone.utc)


class BaseDataProvider(ABC):
    """Abstract base class for data provider adapters.

    Each provider adapter is responsible for:
    1. Connecting to the external data source
    2. Fetching raw data with proper rate limiting and retries
    3. Normalizing data into the shared schema format
    4. Running quality checks on ingested data
    5. Persisting normalized data to the database
    """

    def __init__(self, name: str, enabled: bool = True):
        self.name = name
        self.enabled = enabled

    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to the data provider. Returns True if successful."""
        pass

    @abstractmethod
    async def fetch_daily_bars(
        self,
        symbols: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch daily OHLCV bars for given symbols.

        Returns DataFrame with columns:
            symbol, ts, open, high, low, close, volume, adj_factor, source, received_at
        """
        pass

    @abstractmethod
    async def fetch_fundamentals(
        self,
        symbols: List[str],
    ) -> pd.DataFrame:
        """Fetch fundamental data for given symbols.

        Returns DataFrame with columns:
            issuer_id, fact_name, fact_value, fact_unit, period_end, filed_at, accepted_at, source
        """
        pass

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Check provider health and connectivity."""
        pass

    async def validate_data(self, df: pd.DataFrame) -> List[str]:
        """Run basic quality checks on fetched data.

        Returns list of warning/error messages.
        """
        warnings = []
        if df.empty:
            warnings.append(f"{self.name}: Empty dataset returned")
            return warnings

        # Check for NaN values in critical columns
        critical_cols = ["open", "high", "low", "close"]
        for col in critical_cols:
            if col in df.columns:
                nan_count = df[col].isna().sum()
                if nan_count > 0:
                    warnings.append(f"{self.name}: {nan_count} NaN values in {col}")

        # Check for negative prices
        price_cols = ["open", "high", "low", "close"]
        for col in price_cols:
            if col in df.columns:
                neg_count = (df[col] < 0).sum()
                if neg_count > 0:
                    warnings.append(f"{self.name}: {neg_count} negative values in {col}")

        # Check OHLC consistency
        if all(c in df.columns for c in ["open", "high", "low", "close"]):
            invalid = (df["high"] < df["low"]).sum()
            if invalid > 0:
                warnings.append(f"{self.name}: {invalid} bars where high < low")

        return warnings
