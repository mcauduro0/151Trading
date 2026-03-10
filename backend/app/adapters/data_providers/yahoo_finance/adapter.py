"""Yahoo Finance data provider adapter.

Uses yfinance library for daily OHLCV bars, corporate actions, and basic fundamentals.
No API key required.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import pandas as pd
import yfinance as yf

from app.adapters.data_providers.base import BaseDataProvider, IngestionResult
from app.core.logging import get_logger

logger = get_logger("adapters.yahoo_finance")


class YahooFinanceAdapter(BaseDataProvider):
    """Yahoo Finance data adapter for equities, ETFs, and indices."""

    def __init__(self):
        super().__init__(name="yahoo_finance", enabled=True)

    async def connect(self) -> bool:
        """Yahoo Finance requires no authentication."""
        try:
            test = yf.Ticker("SPY")
            info = test.fast_info
            logger.info("Yahoo Finance connection verified")
            return True
        except Exception as e:
            logger.error("Yahoo Finance connection failed", error=str(e))
            return False

    async def fetch_daily_bars(
        self,
        symbols: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch daily OHLCV bars from Yahoo Finance."""
        result = IngestionResult(provider=self.name, status="running")
        all_bars = []

        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(
                    start=start_date or "2020-01-01",
                    end=end_date,
                    auto_adjust=False,
                )

                if hist.empty:
                    result.warnings.append(f"No data for {symbol}")
                    continue

                # Compute adjustment factor
                if "Adj Close" in hist.columns and "Close" in hist.columns:
                    hist["adj_factor"] = hist["Adj Close"] / hist["Close"]
                else:
                    hist["adj_factor"] = 1.0

                bars = pd.DataFrame({
                    "symbol": symbol,
                    "ts": hist.index.date,
                    "open": hist["Open"].values,
                    "high": hist["High"].values,
                    "low": hist["Low"].values,
                    "close": hist["Close"].values,
                    "volume": hist["Volume"].values,
                    "adj_factor": hist["adj_factor"].values,
                    "source": "yahoo_finance",
                    "received_at": datetime.now(timezone.utc),
                })

                all_bars.append(bars)
                result.records_processed += len(bars)
                logger.info("Fetched bars", symbol=symbol, count=len(bars))

            except Exception as e:
                result.errors.append(f"Error fetching {symbol}: {str(e)}")
                logger.error("Fetch error", symbol=symbol, error=str(e))

        if all_bars:
            df = pd.concat(all_bars, ignore_index=True)
            warnings = await self.validate_data(df)
            result.warnings.extend(warnings)
            result.complete("success" if not result.errors else "partial")
            return df

        result.complete("failed" if result.errors else "success")
        return pd.DataFrame()

    async def fetch_fundamentals(self, symbols: List[str]) -> pd.DataFrame:
        """Fetch basic fundamentals from Yahoo Finance."""
        all_facts = []

        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info or {}

                facts = []
                fundamental_fields = {
                    "marketCap": "market_cap",
                    "trailingPE": "pe_ratio_ttm",
                    "forwardPE": "pe_ratio_forward",
                    "priceToBook": "price_to_book",
                    "dividendYield": "dividend_yield",
                    "trailingEps": "eps_ttm",
                    "revenueGrowth": "revenue_growth",
                    "profitMargins": "profit_margin",
                    "returnOnEquity": "roe",
                    "debtToEquity": "debt_to_equity",
                    "currentRatio": "current_ratio",
                    "beta": "beta",
                }

                now = datetime.now(timezone.utc)
                for yf_key, fact_name in fundamental_fields.items():
                    if yf_key in info and info[yf_key] is not None:
                        facts.append({
                            "symbol": symbol,
                            "fact_name": fact_name,
                            "fact_value": info[yf_key],
                            "fact_unit": "ratio" if "ratio" in fact_name or fact_name in ["beta", "roe"] else "USD",
                            "period_end": now.date(),
                            "filed_at": now,
                            "accepted_at": now,
                            "source": "yahoo_finance",
                        })

                if facts:
                    all_facts.append(pd.DataFrame(facts))

            except Exception as e:
                logger.error("Fundamentals fetch error", symbol=symbol, error=str(e))

        return pd.concat(all_facts, ignore_index=True) if all_facts else pd.DataFrame()

    async def health_check(self) -> Dict[str, Any]:
        """Check Yahoo Finance connectivity."""
        try:
            ticker = yf.Ticker("SPY")
            hist = ticker.history(period="1d")
            return {
                "provider": self.name,
                "status": "healthy" if not hist.empty else "degraded",
                "last_check": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            return {
                "provider": self.name,
                "status": "unhealthy",
                "error": str(e),
                "last_check": datetime.now(timezone.utc).isoformat(),
            }
