"""Polygon.io data provider adapter.

Provides real-time and historical market data for stocks, options, forex,
crypto, and indices. Uses the official polygon-api-client SDK.
"""

from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional
import pandas as pd

from app.adapters.data_providers.base import BaseDataProvider
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger("adapters.polygon")


class PolygonAdapter(BaseDataProvider):
    """Polygon.io data adapter for comprehensive market data."""

    def __init__(self):
        super().__init__(name="polygon", enabled=settings.polygon_enabled)
        self.api_key = settings.polygon_api_key
        self._client = None

    async def connect(self) -> bool:
        """Initialize Polygon REST client."""
        if not self.api_key:
            logger.warning("Polygon API key not configured")
            return False
        try:
            from polygon import RESTClient
            self._client = RESTClient(api_key=self.api_key)
            # Test connection
            aggs = list(self._client.list_aggs("SPY", 1, "day",
                                                (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d"),
                                                datetime.now().strftime("%Y-%m-%d"),
                                                limit=5))
            logger.info("Polygon connection verified", test_bars=len(aggs))
            return True
        except Exception as e:
            logger.error("Polygon connection failed", error=str(e))
            return False

    async def fetch_daily_bars(
        self,
        symbols: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch daily OHLCV bars from Polygon."""
        if not self._client:
            await self.connect()
        if not self._client:
            return pd.DataFrame()

        all_bars = []
        start = start_date or "2020-01-01"
        end = end_date or datetime.now().strftime("%Y-%m-%d")

        for symbol in symbols:
            try:
                aggs = list(self._client.list_aggs(
                    symbol, 1, "day", start, end, limit=50000
                ))

                if aggs:
                    bars = pd.DataFrame([{
                        "symbol": symbol,
                        "ts": datetime.fromtimestamp(a.timestamp / 1000, tz=timezone.utc).date(),
                        "open": a.open,
                        "high": a.high,
                        "low": a.low,
                        "close": a.close,
                        "volume": a.volume,
                        "adj_factor": 1.0,
                        "source": "polygon",
                        "received_at": datetime.now(timezone.utc),
                    } for a in aggs])

                    all_bars.append(bars)
                    logger.info("Fetched Polygon bars", symbol=symbol, count=len(bars))

            except Exception as e:
                logger.error("Polygon bars error", symbol=symbol, error=str(e))

        return pd.concat(all_bars, ignore_index=True) if all_bars else pd.DataFrame()

    async def fetch_options_chain(
        self,
        underlying: str,
        expiry_from: Optional[str] = None,
        expiry_to: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch options chain data from Polygon."""
        if not self._client:
            await self.connect()
        if not self._client:
            return pd.DataFrame()

        try:
            contracts = list(self._client.list_options_contracts(
                underlying_ticker=underlying,
                expiration_date_gte=expiry_from,
                expiration_date_lte=expiry_to,
                limit=1000,
            ))

            if contracts:
                return pd.DataFrame([{
                    "underlying": underlying,
                    "ticker": c.ticker,
                    "expiry": c.expiration_date,
                    "strike": c.strike_price,
                    "right": c.contract_type,
                    "style": c.exercise_style or "american",
                } for c in contracts])

        except Exception as e:
            logger.error("Polygon options error", underlying=underlying, error=str(e))

        return pd.DataFrame()

    async def fetch_fundamentals(self, symbols: List[str]) -> pd.DataFrame:
        """Polygon fundamentals are limited - use FMP as primary source."""
        return pd.DataFrame()

    async def health_check(self) -> Dict[str, Any]:
        """Check Polygon API connectivity."""
        try:
            if not self._client:
                await self.connect()
            if self._client:
                aggs = list(self._client.list_aggs("SPY", 1, "day",
                                                    (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d"),
                                                    datetime.now().strftime("%Y-%m-%d"), limit=1))
                return {"provider": self.name, "status": "healthy",
                        "last_check": datetime.now(timezone.utc).isoformat()}
            return {"provider": self.name, "status": "not_configured"}
        except Exception as e:
            return {"provider": self.name, "status": "unhealthy", "error": str(e)}
