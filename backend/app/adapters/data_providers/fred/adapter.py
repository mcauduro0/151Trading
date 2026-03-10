"""FRED (Federal Reserve Economic Data) provider adapter.

Provides macro-economic time series: GDP, CPI, unemployment, yield curves,
Fed Funds rate, and hundreds of other economic indicators.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import pandas as pd

from app.adapters.data_providers.base import BaseDataProvider
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger("adapters.fred")

# Key FRED series for the trading system
CORE_SERIES = {
    # Yield curve
    "DGS1MO": "Treasury 1-Month",
    "DGS3MO": "Treasury 3-Month",
    "DGS6MO": "Treasury 6-Month",
    "DGS1": "Treasury 1-Year",
    "DGS2": "Treasury 2-Year",
    "DGS5": "Treasury 5-Year",
    "DGS7": "Treasury 7-Year",
    "DGS10": "Treasury 10-Year",
    "DGS20": "Treasury 20-Year",
    "DGS30": "Treasury 30-Year",
    # Macro indicators
    "GDP": "Gross Domestic Product",
    "CPIAUCSL": "Consumer Price Index",
    "UNRATE": "Unemployment Rate",
    "FEDFUNDS": "Federal Funds Rate",
    "T10Y2Y": "10Y-2Y Treasury Spread",
    "T10Y3M": "10Y-3M Treasury Spread",
    "VIXCLS": "VIX Close",
    "DTWEXBGS": "Trade Weighted USD Index",
    "BAMLH0A0HYM2": "High Yield OAS",
    "BAMLC0A4CBBB": "BBB Corporate OAS",
    "DCOILWTICO": "WTI Crude Oil",
    "GOLDAMGBD228NLBM": "Gold Price London",
    "M2SL": "M2 Money Supply",
    "WALCL": "Fed Balance Sheet",
}


class FREDAdapter(BaseDataProvider):
    """FRED data adapter for macro-economic data."""

    def __init__(self):
        super().__init__(name="fred", enabled=settings.fred_enabled)
        self.api_key = settings.fred_api_key
        self._fred = None

    async def connect(self) -> bool:
        """Initialize FRED API connection."""
        if not self.api_key:
            logger.warning("FRED API key not configured")
            return False
        try:
            from fredapi import Fred
            self._fred = Fred(api_key=self.api_key)
            # Test connection
            self._fred.get_series("DGS10", observation_start="2025-01-01")
            logger.info("FRED connection verified")
            return True
        except Exception as e:
            logger.error("FRED connection failed", error=str(e))
            return False

    async def fetch_daily_bars(
        self,
        symbols: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """FRED doesn't provide bars - returns empty DataFrame."""
        return pd.DataFrame()

    async def fetch_macro_series(
        self,
        series_ids: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch macro-economic time series from FRED."""
        if not self._fred:
            await self.connect()
        if not self._fred:
            return pd.DataFrame()

        target_series = series_ids or list(CORE_SERIES.keys())
        all_obs = []

        for series_id in target_series:
            try:
                data = self._fred.get_series(
                    series_id,
                    observation_start=start_date or "2000-01-01",
                    observation_end=end_date,
                )

                if data is not None and not data.empty:
                    obs = pd.DataFrame({
                        "series_id": series_id,
                        "obs_date": data.index.date,
                        "value": data.values,
                        "vintage_date": datetime.now(timezone.utc).date(),
                        "released_at": None,
                        "received_at": datetime.now(timezone.utc),
                    })
                    all_obs.append(obs)
                    logger.info("Fetched FRED series", series_id=series_id, count=len(obs))

            except Exception as e:
                logger.error("FRED fetch error", series_id=series_id, error=str(e))

        return pd.concat(all_obs, ignore_index=True) if all_obs else pd.DataFrame()

    async def fetch_fundamentals(self, symbols: List[str]) -> pd.DataFrame:
        """FRED doesn't provide company fundamentals."""
        return pd.DataFrame()

    async def health_check(self) -> Dict[str, Any]:
        """Check FRED API connectivity."""
        try:
            if not self._fred:
                await self.connect()
            if self._fred:
                self._fred.get_series("DGS10", observation_start="2025-01-01")
                return {"provider": self.name, "status": "healthy",
                        "last_check": datetime.now(timezone.utc).isoformat()}
            return {"provider": self.name, "status": "not_configured"}
        except Exception as e:
            return {"provider": self.name, "status": "unhealthy", "error": str(e)}
