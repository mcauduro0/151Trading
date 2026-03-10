"""FMP (Financial Modeling Prep) data provider adapter.

Primary source for company fundamentals, financial statements, ratios,
earnings, and company profiles. Ultimate tier access.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import pandas as pd
import httpx

from app.adapters.data_providers.base import BaseDataProvider
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger("adapters.fmp")

FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"


class FMPAdapter(BaseDataProvider):
    """FMP data adapter for fundamentals, financials, and company data."""

    def __init__(self):
        super().__init__(name="fmp", enabled=settings.fmp_enabled)
        self.api_key = settings.fmp_api_key
        self.base_url = FMP_BASE_URL

    async def connect(self) -> bool:
        """Verify FMP API connectivity."""
        if not self.api_key:
            logger.warning("FMP API key not configured")
            return False
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{self.base_url}/profile/AAPL",
                    params={"apikey": self.api_key},
                    timeout=10,
                )
                resp.raise_for_status()
                logger.info("FMP connection verified")
                return True
        except Exception as e:
            logger.error("FMP connection failed", error=str(e))
            return False

    async def _get(self, endpoint: str, params: dict = None) -> Any:
        """Make authenticated GET request to FMP API."""
        params = params or {}
        params["apikey"] = self.api_key
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{self.base_url}/{endpoint}",
                params=params,
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()

    async def fetch_daily_bars(
        self,
        symbols: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch daily historical prices from FMP."""
        all_bars = []

        for symbol in symbols:
            try:
                params = {}
                if start_date:
                    params["from"] = start_date
                if end_date:
                    params["to"] = end_date

                data = await self._get(f"historical-price-full/{symbol}", params)

                if "historical" in data:
                    hist = data["historical"]
                    bars = pd.DataFrame(hist)
                    bars["symbol"] = symbol
                    bars["ts"] = pd.to_datetime(bars["date"]).dt.date
                    bars["adj_factor"] = bars.get("adjClose", bars["close"]) / bars["close"]
                    bars["source"] = "fmp"
                    bars["received_at"] = datetime.now(timezone.utc)

                    bars = bars.rename(columns={
                        "open": "open", "high": "high", "low": "low",
                        "close": "close", "volume": "volume",
                    })

                    all_bars.append(bars[["symbol", "ts", "open", "high", "low", "close",
                                          "volume", "adj_factor", "source", "received_at"]])
                    logger.info("Fetched FMP bars", symbol=symbol, count=len(bars))

            except Exception as e:
                logger.error("FMP bars fetch error", symbol=symbol, error=str(e))

        return pd.concat(all_bars, ignore_index=True) if all_bars else pd.DataFrame()

    async def fetch_fundamentals(self, symbols: List[str]) -> pd.DataFrame:
        """Fetch comprehensive fundamentals from FMP."""
        all_facts = []

        for symbol in symbols:
            try:
                # Income statement
                income = await self._get(f"income-statement/{symbol}", {"period": "annual", "limit": 10})
                # Balance sheet
                balance = await self._get(f"balance-sheet-statement/{symbol}", {"period": "annual", "limit": 10})
                # Ratios
                ratios = await self._get(f"ratios/{symbol}", {"period": "annual", "limit": 10})
                # Profile
                profile = await self._get(f"profile/{symbol}")

                now = datetime.now(timezone.utc)

                # Process income statement
                for stmt in (income or []):
                    period_end = stmt.get("date", "")
                    for key, value in stmt.items():
                        if isinstance(value, (int, float)) and key not in ("date", "symbol", "cik"):
                            all_facts.append({
                                "symbol": symbol,
                                "fact_name": f"income_{key}",
                                "fact_value": value,
                                "fact_unit": "USD",
                                "period_end": period_end,
                                "filed_at": stmt.get("fillingDate", now),
                                "accepted_at": stmt.get("acceptedDate", now),
                                "source": "fmp",
                            })

                # Process balance sheet
                for stmt in (balance or []):
                    period_end = stmt.get("date", "")
                    for key, value in stmt.items():
                        if isinstance(value, (int, float)) and key not in ("date", "symbol", "cik"):
                            all_facts.append({
                                "symbol": symbol,
                                "fact_name": f"balance_{key}",
                                "fact_value": value,
                                "fact_unit": "USD",
                                "period_end": period_end,
                                "filed_at": stmt.get("fillingDate", now),
                                "accepted_at": stmt.get("acceptedDate", now),
                                "source": "fmp",
                            })

                # Process ratios
                for r in (ratios or []):
                    period_end = r.get("date", "")
                    for key, value in r.items():
                        if isinstance(value, (int, float)) and key not in ("date", "symbol"):
                            all_facts.append({
                                "symbol": symbol,
                                "fact_name": f"ratio_{key}",
                                "fact_value": value,
                                "fact_unit": "ratio",
                                "period_end": period_end,
                                "filed_at": now,
                                "accepted_at": now,
                                "source": "fmp",
                            })

                logger.info("Fetched FMP fundamentals", symbol=symbol)

            except Exception as e:
                logger.error("FMP fundamentals error", symbol=symbol, error=str(e))

        return pd.DataFrame(all_facts) if all_facts else pd.DataFrame()

    async def fetch_company_profile(self, symbol: str) -> Dict[str, Any]:
        """Fetch company profile."""
        data = await self._get(f"profile/{symbol}")
        return data[0] if data else {}

    async def fetch_earnings(self, symbol: str) -> List[Dict]:
        """Fetch earnings history."""
        return await self._get(f"historical/earning_calendar/{symbol}") or []

    async def health_check(self) -> Dict[str, Any]:
        """Check FMP API connectivity."""
        try:
            data = await self._get("profile/AAPL")
            return {"provider": self.name, "status": "healthy" if data else "degraded",
                    "last_check": datetime.now(timezone.utc).isoformat()}
        except Exception as e:
            return {"provider": self.name, "status": "unhealthy", "error": str(e)}
