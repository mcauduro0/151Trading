"""B3 and Anbima data provider adapter for Brazilian market data.

B3 (Brasil Bolsa Balcão) provides equity, options, futures, and fixed income data.
Anbima provides Brazilian fixed income benchmarks, IMA indices, and fund data.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import pandas as pd
import httpx

from app.adapters.data_providers.base import BaseDataProvider
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger("adapters.b3_anbima")

# B3 public data endpoints
B3_MARKET_DATA_URL = "https://arquivos.b3.com.br/api/download/requestname"
# Anbima public endpoints
ANBIMA_IMA_URL = "https://www.anbima.com.br/informacoes/ima"


class B3AnbimaAdapter(BaseDataProvider):
    """B3 and Anbima adapter for Brazilian market data."""

    def __init__(self):
        super().__init__(name="b3_anbima", enabled=settings.b3_enabled)

    async def connect(self) -> bool:
        """Verify B3/Anbima data accessibility."""
        try:
            async with httpx.AsyncClient() as client:
                # Test B3 public endpoint
                resp = await client.get(
                    "https://sistemaswebb3-listados.b3.com.br/indexProxy/indexCall/GetPortfolioDay/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjIwLCJpbmRleCI6IklCT1YiLCJzZWdtZW50IjoiMSJ9",
                    timeout=15,
                )
                logger.info("B3/Anbima connection verified", status=resp.status_code)
                return resp.status_code == 200
        except Exception as e:
            logger.error("B3/Anbima connection failed", error=str(e))
            return False

    async def fetch_daily_bars(
        self,
        symbols: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch daily bars for Brazilian equities via B3 public data.

        For Brazilian equities, we use Yahoo Finance with .SA suffix as primary,
        and B3 public files as supplementary source.
        """
        import yfinance as yf

        all_bars = []
        for symbol in symbols:
            try:
                # Add .SA suffix for B3 tickers if not present
                yf_symbol = f"{symbol}.SA" if not symbol.endswith(".SA") else symbol
                ticker = yf.Ticker(yf_symbol)
                hist = ticker.history(
                    start=start_date or "2020-01-01",
                    end=end_date,
                    auto_adjust=False,
                )

                if not hist.empty:
                    if "Adj Close" in hist.columns:
                        adj_factor = hist["Adj Close"] / hist["Close"]
                    else:
                        adj_factor = 1.0

                    bars = pd.DataFrame({
                        "symbol": symbol,
                        "ts": hist.index.date,
                        "open": hist["Open"].values,
                        "high": hist["High"].values,
                        "low": hist["Low"].values,
                        "close": hist["Close"].values,
                        "volume": hist["Volume"].values,
                        "adj_factor": adj_factor,
                        "source": "b3_yahoo",
                        "received_at": datetime.now(timezone.utc),
                    })
                    all_bars.append(bars)
                    logger.info("Fetched B3 bars", symbol=symbol, count=len(bars))

            except Exception as e:
                logger.error("B3 bars error", symbol=symbol, error=str(e))

        return pd.concat(all_bars, ignore_index=True) if all_bars else pd.DataFrame()

    async def fetch_ibovespa_composition(self) -> pd.DataFrame:
        """Fetch current Ibovespa index composition."""
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    "https://sistemaswebb3-listados.b3.com.br/indexProxy/indexCall/GetPortfolioDay/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjIwMCwiaW5kZXgiOiJJQk9WIiwic2VnbWVudCI6IjEifQ==",
                    timeout=15,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    results = data.get("results", [])
                    return pd.DataFrame(results) if results else pd.DataFrame()
        except Exception as e:
            logger.error("Ibovespa composition error", error=str(e))
        return pd.DataFrame()

    async def fetch_fundamentals(self, symbols: List[str]) -> pd.DataFrame:
        """B3 fundamentals are limited - use FMP or Fiscal AI."""
        return pd.DataFrame()

    async def health_check(self) -> Dict[str, Any]:
        """Check B3/Anbima data accessibility."""
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    "https://sistemaswebb3-listados.b3.com.br/indexProxy/indexCall/GetPortfolioDay/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjIwLCJpbmRleCI6IklCT1YiLCJzZWdtZW50IjoiMSJ9",
                    timeout=10,
                )
                return {"provider": self.name, "status": "healthy" if resp.status_code == 200 else "degraded",
                        "last_check": datetime.now(timezone.utc).isoformat()}
        except Exception as e:
            return {"provider": self.name, "status": "unhealthy", "error": str(e)}
