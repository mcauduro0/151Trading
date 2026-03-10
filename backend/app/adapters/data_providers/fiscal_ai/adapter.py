"""Fiscal AI (bluefiscal) data provider adapter.

Provides high-quality financial data: income statements, balance sheets,
cash flow statements, ratios, segments, and KPIs. As-reported and standardized.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import pandas as pd
import httpx

from app.adapters.data_providers.base import BaseDataProvider
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger("adapters.fiscal_ai")

FISCAL_AI_BASE_URL = "https://api.fiscal.ai/v1"


class FiscalAIAdapter(BaseDataProvider):
    """Fiscal AI adapter for detailed financial data."""

    def __init__(self):
        super().__init__(name="fiscal_ai", enabled=settings.fiscal_ai_enabled)
        self.api_key = settings.fiscal_ai_api_key

    async def connect(self) -> bool:
        """Verify Fiscal AI API connectivity."""
        if not self.api_key:
            logger.warning("Fiscal AI API key not configured")
            return False
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{FISCAL_AI_BASE_URL}/companies-list",
                    params={"apiKey": self.api_key},
                    timeout=15,
                )
                resp.raise_for_status()
                logger.info("Fiscal AI connection verified")
                return True
        except Exception as e:
            logger.error("Fiscal AI connection failed", error=str(e))
            return False

    async def _get(self, endpoint: str, params: dict = None) -> Any:
        """Make authenticated GET request to Fiscal AI."""
        params = params or {}
        params["apiKey"] = self.api_key
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{FISCAL_AI_BASE_URL}/{endpoint}",
                params=params,
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()

    def _make_company_key(self, symbol: str, exchange: str = "NASDAQ") -> str:
        """Create Fiscal AI companyKey format: EXCHANGE_TICKER."""
        return f"{exchange}_{symbol}"

    async def fetch_daily_bars(self, symbols: List[str], **kwargs) -> pd.DataFrame:
        """Fiscal AI doesn't provide price bars as primary data."""
        return pd.DataFrame()

    async def fetch_fundamentals(self, symbols: List[str]) -> pd.DataFrame:
        """Fetch comprehensive as-reported fundamentals from Fiscal AI."""
        all_facts = []

        for symbol in symbols:
            try:
                # Try common exchanges
                for exchange in ["NASDAQ", "NYSE", "AMEX"]:
                    company_key = self._make_company_key(symbol, exchange)

                    try:
                        # Income statement (as-reported)
                        income = await self._get(
                            "company/financials/income-statement/as-reported",
                            {"companyKey": company_key}
                        )
                        if income:
                            for stmt in income if isinstance(income, list) else [income]:
                                period = stmt.get("periodEnd", "")
                                for key, value in stmt.items():
                                    if isinstance(value, (int, float)):
                                        all_facts.append({
                                            "symbol": symbol,
                                            "fact_name": f"fiscal_income_{key}",
                                            "fact_value": value,
                                            "fact_unit": "USD",
                                            "period_end": period,
                                            "filed_at": datetime.now(timezone.utc),
                                            "accepted_at": datetime.now(timezone.utc),
                                            "source": "fiscal_ai",
                                        })

                        # Balance sheet (as-reported)
                        balance = await self._get(
                            "company/financials/balance-sheet/as-reported",
                            {"companyKey": company_key}
                        )
                        if balance:
                            for stmt in balance if isinstance(balance, list) else [balance]:
                                period = stmt.get("periodEnd", "")
                                for key, value in stmt.items():
                                    if isinstance(value, (int, float)):
                                        all_facts.append({
                                            "symbol": symbol,
                                            "fact_name": f"fiscal_balance_{key}",
                                            "fact_value": value,
                                            "fact_unit": "USD",
                                            "period_end": period,
                                            "filed_at": datetime.now(timezone.utc),
                                            "accepted_at": datetime.now(timezone.utc),
                                            "source": "fiscal_ai",
                                        })

                        # Ratios
                        ratios = await self._get(
                            "company/ratios",
                            {"companyKey": company_key}
                        )
                        if ratios:
                            for r in ratios if isinstance(ratios, list) else [ratios]:
                                period = r.get("periodEnd", "")
                                for key, value in r.items():
                                    if isinstance(value, (int, float)):
                                        all_facts.append({
                                            "symbol": symbol,
                                            "fact_name": f"fiscal_ratio_{key}",
                                            "fact_value": value,
                                            "fact_unit": "ratio",
                                            "period_end": period,
                                            "filed_at": datetime.now(timezone.utc),
                                            "accepted_at": datetime.now(timezone.utc),
                                            "source": "fiscal_ai",
                                        })

                        if all_facts:
                            break  # Found data, no need to try other exchanges

                    except httpx.HTTPStatusError as e:
                        if e.response.status_code == 404:
                            continue  # Try next exchange
                        raise

                logger.info("Fetched Fiscal AI fundamentals", symbol=symbol,
                           facts=len([f for f in all_facts if f["symbol"] == symbol]))

            except Exception as e:
                logger.error("Fiscal AI error", symbol=symbol, error=str(e))

        return pd.DataFrame(all_facts) if all_facts else pd.DataFrame()

    async def fetch_company_profile(self, symbol: str) -> Dict[str, Any]:
        """Fetch company profile from Fiscal AI."""
        for exchange in ["NASDAQ", "NYSE", "AMEX"]:
            try:
                company_key = self._make_company_key(symbol, exchange)
                data = await self._get("company/profile", {"companyKey": company_key})
                if data:
                    return data
            except Exception:
                continue
        return {}

    async def health_check(self) -> Dict[str, Any]:
        """Check Fiscal AI API connectivity."""
        try:
            data = await self._get("companies-list")
            return {"provider": self.name, "status": "healthy" if data else "degraded",
                    "last_check": datetime.now(timezone.utc).isoformat()}
        except Exception as e:
            return {"provider": self.name, "status": "unhealthy", "error": str(e)}
