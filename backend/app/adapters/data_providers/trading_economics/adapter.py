"""Trading Economics data provider adapter.

Provides macro-economic data, economic calendar, indicators, and
forecasts for 196 countries.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import pandas as pd
import httpx

from app.adapters.data_providers.base import BaseDataProvider
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger("adapters.trading_economics")

TE_BASE_URL = "https://api.tradingeconomics.com"


class TradingEconomicsAdapter(BaseDataProvider):
    """Trading Economics adapter for global macro data."""

    def __init__(self):
        super().__init__(name="trading_economics", enabled=settings.trading_economics_enabled)
        self.client_key = settings.trading_economics_client_key
        self.secret_key = settings.trading_economics_secret_key

    async def connect(self) -> bool:
        """Verify Trading Economics API connectivity."""
        if not self.client_key:
            logger.warning("Trading Economics credentials not configured")
            return False
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{TE_BASE_URL}/calendar",
                    params={"c": f"{self.client_key}:{self.secret_key}", "f": "json"},
                    timeout=15,
                )
                resp.raise_for_status()
                logger.info("Trading Economics connection verified")
                return True
        except Exception as e:
            logger.error("Trading Economics connection failed", error=str(e))
            return False

    async def _get(self, endpoint: str, params: dict = None) -> Any:
        """Make authenticated GET request."""
        params = params or {}
        params["c"] = f"{self.client_key}:{self.secret_key}"
        params["f"] = "json"
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{TE_BASE_URL}/{endpoint}", params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()

    async def fetch_daily_bars(self, symbols: List[str], **kwargs) -> pd.DataFrame:
        """Trading Economics doesn't provide standard bars."""
        return pd.DataFrame()

    async def fetch_fundamentals(self, symbols: List[str]) -> pd.DataFrame:
        """Trading Economics doesn't provide company fundamentals."""
        return pd.DataFrame()

    async def fetch_economic_calendar(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        country: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch economic calendar events."""
        try:
            endpoint = "calendar"
            if start_date and end_date:
                endpoint = f"calendar/country/All/{start_date}/{end_date}"
            elif country:
                endpoint = f"calendar/country/{country}"

            data = await self._get(endpoint)

            if data:
                df = pd.DataFrame(data)
                df["received_at"] = datetime.now(timezone.utc)
                df["source"] = "trading_economics"
                logger.info("Fetched economic calendar", events=len(df))
                return df

        except Exception as e:
            logger.error("TE calendar error", error=str(e))

        return pd.DataFrame()

    async def fetch_indicators(
        self,
        country: str = "united states",
        indicators: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Fetch economic indicators for a country."""
        try:
            data = await self._get(f"country/{country}")
            if data:
                df = pd.DataFrame(data)
                df["received_at"] = datetime.now(timezone.utc)
                df["source"] = "trading_economics"
                if indicators:
                    df = df[df["Category"].isin(indicators)]
                return df
        except Exception as e:
            logger.error("TE indicators error", country=country, error=str(e))
        return pd.DataFrame()

    async def health_check(self) -> Dict[str, Any]:
        """Check Trading Economics API connectivity."""
        try:
            data = await self._get("calendar")
            return {"provider": self.name, "status": "healthy" if data else "degraded",
                    "last_check": datetime.now(timezone.utc).isoformat()}
        except Exception as e:
            return {"provider": self.name, "status": "unhealthy", "error": str(e)}
