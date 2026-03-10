"""
Data Ingestion Pipeline — Orchestrates all data provider adapters.
Loads configuration from .env, validates API keys, and provides
a unified interface for pulling market data, fundamentals, and macro data.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
_env_path = Path(__file__).resolve().parents[4] / ".env"
load_dotenv(_env_path)

logger = logging.getLogger("data_ingestion")


class DataIngestionConfig:
    """Centralized configuration for all data providers."""

    def __init__(self):
        # Yahoo Finance
        self.yahoo_enabled = os.getenv("YAHOO_FINANCE_ENABLED", "true").lower() == "true"

        # FRED
        self.fred_api_key = os.getenv("FRED_API_KEY", "")
        self.fred_enabled = os.getenv("FRED_ENABLED", "true").lower() == "true" and bool(self.fred_api_key)

        # FMP
        self.fmp_api_key = os.getenv("FMP_API_KEY", "")
        self.fmp_enabled = os.getenv("FMP_ENABLED", "true").lower() == "true" and bool(self.fmp_api_key)

        # Polygon
        self.polygon_api_key = os.getenv("POLYGON_API_KEY", "")
        self.polygon_enabled = os.getenv("POLYGON_ENABLED", "true").lower() == "true" and bool(self.polygon_api_key)

        # Reddit
        self.reddit_client_id = os.getenv("REDDIT_CLIENT_ID", "")
        self.reddit_client_secret = os.getenv("REDDIT_CLIENT_SECRET", "")
        self.reddit_enabled = os.getenv("REDDIT_ENABLED", "false").lower() == "true"

        # Trading Economics
        self.te_client_key = os.getenv("TRADING_ECONOMICS_CLIENT_KEY", "")
        self.te_secret_key = os.getenv("TRADING_ECONOMICS_SECRET_KEY", "")
        self.te_enabled = os.getenv("TRADING_ECONOMICS_ENABLED", "true").lower() == "true" and bool(self.te_client_key)

        # B3 / Anbima
        self.b3_enabled = os.getenv("B3_ENABLED", "true").lower() == "true"
        self.anbima_enabled = os.getenv("ANBIMA_ENABLED", "true").lower() == "true"

        # Fiscal AI
        self.fiscal_ai_api_key = os.getenv("FISCAL_AI_API_KEY", "")
        self.fiscal_ai_base_url = os.getenv("FISCAL_AI_BASE_URL", "https://api.fiscal.ai/v1")
        self.fiscal_ai_enabled = os.getenv("FISCAL_AI_ENABLED", "true").lower() == "true" and bool(self.fiscal_ai_api_key)

    def status_report(self) -> dict:
        """Return status of all providers."""
        return {
            "yahoo_finance": {"enabled": self.yahoo_enabled, "has_key": True},
            "fred": {"enabled": self.fred_enabled, "has_key": bool(self.fred_api_key)},
            "fmp": {"enabled": self.fmp_enabled, "has_key": bool(self.fmp_api_key)},
            "polygon": {"enabled": self.polygon_enabled, "has_key": bool(self.polygon_api_key)},
            "reddit": {"enabled": self.reddit_enabled, "has_key": bool(self.reddit_client_id)},
            "trading_economics": {"enabled": self.te_enabled, "has_key": bool(self.te_client_key)},
            "b3_anbima": {"enabled": self.b3_enabled or self.anbima_enabled, "has_key": True},
            "fiscal_ai": {"enabled": self.fiscal_ai_enabled, "has_key": bool(self.fiscal_ai_api_key)},
        }


class YahooFinanceClient:
    """Production Yahoo Finance client using yfinance."""

    def __init__(self):
        import yfinance as yf
        self.yf = yf

    def get_prices(self, tickers: list[str], period: str = "2y", interval: str = "1d"):
        """Fetch OHLCV data for multiple tickers."""
        import pandas as pd
        data = self.yf.download(tickers, period=period, interval=interval, group_by="ticker", threads=True)
        return data

    def get_ticker_info(self, ticker: str) -> dict:
        """Get fundamental info for a single ticker."""
        t = self.yf.Ticker(ticker)
        return t.info

    def get_financials(self, ticker: str):
        """Get financial statements."""
        t = self.yf.Ticker(ticker)
        return {
            "income_stmt": t.income_stmt,
            "balance_sheet": t.balance_sheet,
            "cashflow": t.cashflow,
        }


class FREDClient:
    """Production FRED client using fredapi."""

    def __init__(self, api_key: str):
        from fredapi import Fred
        self.fred = Fred(api_key=api_key)

    def get_series(self, series_id: str, start_date: Optional[str] = None, end_date: Optional[str] = None):
        """Fetch a FRED time series."""
        kwargs = {}
        if start_date:
            kwargs["observation_start"] = start_date
        if end_date:
            kwargs["observation_end"] = end_date
        return self.fred.get_series(series_id, **kwargs)

    def get_macro_indicators(self):
        """Fetch key macro indicators for regime detection."""
        indicators = {
            "GDP": "GDP",
            "CPI": "CPIAUCSL",
            "UNRATE": "UNRATE",
            "FEDFUNDS": "FEDFUNDS",
            "T10Y2Y": "T10Y2Y",
            "T10YIE": "T10YIE",
            "VIXCLS": "VIXCLS",
            "SP500": "SP500",
            "BAMLH0A0HYM2": "BAMLH0A0HYM2",  # HY spread
            "DGS10": "DGS10",
            "DGS2": "DGS2",
            "DGS5": "DGS5",
            "DGS30": "DGS30",
        }
        results = {}
        for name, series_id in indicators.items():
            try:
                results[name] = self.fred.get_series(series_id)
            except Exception as e:
                logger.warning(f"Failed to fetch FRED series {series_id}: {e}")
        return results


class FMPClient:
    """Production FMP client for fundamental data."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://financialmodelingprep.com/api/v3"

    def _get(self, endpoint: str, params: Optional[dict] = None):
        import requests
        params = params or {}
        params["apikey"] = self.api_key
        url = f"{self.base_url}/{endpoint}"
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def get_financial_ratios(self, ticker: str, period: str = "annual", limit: int = 10):
        return self._get(f"ratios/{ticker}", {"period": period, "limit": limit})

    def get_key_metrics(self, ticker: str, period: str = "annual", limit: int = 10):
        return self._get(f"key-metrics/{ticker}", {"period": period, "limit": limit})

    def get_income_statement(self, ticker: str, period: str = "annual", limit: int = 10):
        return self._get(f"income-statement/{ticker}", {"period": period, "limit": limit})

    def get_balance_sheet(self, ticker: str, period: str = "annual", limit: int = 10):
        return self._get(f"balance-sheet-statement/{ticker}", {"period": period, "limit": limit})

    def get_cash_flow(self, ticker: str, period: str = "annual", limit: int = 10):
        return self._get(f"cash-flow-statement/{ticker}", {"period": period, "limit": limit})

    def get_profile(self, ticker: str):
        return self._get(f"profile/{ticker}")

    def get_stock_screener(self, market_cap_min: int = 10_000_000_000, limit: int = 500):
        """Screen for large-cap US stocks."""
        return self._get("stock-screener", {
            "marketCapMoreThan": market_cap_min,
            "exchange": "NYSE,NASDAQ",
            "isActivelyTrading": True,
            "limit": limit,
        })

    def get_sp500_constituents(self):
        """Get S&P 500 constituent list."""
        return self._get("sp500_constituent")

    def get_historical_price(self, ticker: str, from_date: str = None, to_date: str = None):
        params = {}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        return self._get(f"historical-price-full/{ticker}", params)


class TradingEconomicsClient:
    """Production Trading Economics client."""

    def __init__(self, client_key: str, secret_key: str):
        self.client_key = client_key
        self.secret_key = secret_key
        self.base_url = "https://api.tradingeconomics.com"

    def _get(self, endpoint: str, params: Optional[dict] = None):
        import requests
        params = params or {}
        params["c"] = f"{self.client_key}:{self.secret_key}"
        url = f"{self.base_url}/{endpoint}"
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def get_indicators(self, country: str = "united states"):
        return self._get(f"country/{country}")

    def get_calendar(self, start_date: str = None, end_date: str = None):
        params = {}
        if start_date:
            params["d1"] = start_date
        if end_date:
            params["d2"] = end_date
        return self._get("calendar", params)

    def get_markets(self, market_type: str = "index"):
        return self._get(f"markets/{market_type}")


class DataIngestionPipeline:
    """Main orchestrator for all data ingestion."""

    def __init__(self):
        self.config = DataIngestionConfig()
        self._yahoo = None
        self._fred = None
        self._fmp = None
        self._te = None

    @property
    def yahoo(self) -> YahooFinanceClient:
        if self._yahoo is None and self.config.yahoo_enabled:
            self._yahoo = YahooFinanceClient()
        return self._yahoo

    @property
    def fred(self) -> Optional[FREDClient]:
        if self._fred is None and self.config.fred_enabled:
            self._fred = FREDClient(self.config.fred_api_key)
        return self._fred

    @property
    def fmp(self) -> Optional[FMPClient]:
        if self._fmp is None and self.config.fmp_enabled:
            self._fmp = FMPClient(self.config.fmp_api_key)
        return self._fmp

    @property
    def te(self) -> Optional[TradingEconomicsClient]:
        if self._te is None and self.config.te_enabled:
            self._te = TradingEconomicsClient(self.config.te_client_key, self.config.te_secret_key)
        return self._te

    def get_equity_universe(self, min_market_cap: int = 10_000_000_000) -> list[str]:
        """Get the equity universe (S&P 500 constituents)."""
        if self.fmp:
            try:
                constituents = self.fmp.get_sp500_constituents()
                return [c["symbol"] for c in constituents]
            except Exception as e:
                logger.warning(f"FMP SP500 fetch failed: {e}, falling back to Yahoo")

        # Fallback: use a static list of major tickers
        return [
            "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "BRK-B", "TSLA", "UNH", "XOM",
            "JNJ", "JPM", "V", "PG", "MA", "HD", "CVX", "MRK", "ABBV", "LLY",
            "PEP", "KO", "COST", "AVGO", "WMT", "MCD", "CSCO", "TMO", "ACN", "ABT",
            "DHR", "NEE", "LIN", "TXN", "PM", "UNP", "RTX", "LOW", "HON", "AMGN",
            "COP", "BMY", "QCOM", "ELV", "SBUX", "GS", "CAT", "BLK", "INTU", "DE",
        ]

    def get_price_data(self, tickers: list[str], period: str = "2y") -> "pd.DataFrame":
        """Fetch price data for a list of tickers."""
        return self.yahoo.get_prices(tickers, period=period)

    def get_fundamental_data(self, ticker: str) -> dict:
        """Fetch fundamental data from FMP (preferred) or Yahoo."""
        if self.fmp:
            try:
                return {
                    "ratios": self.fmp.get_financial_ratios(ticker),
                    "key_metrics": self.fmp.get_key_metrics(ticker),
                    "income_stmt": self.fmp.get_income_statement(ticker),
                    "balance_sheet": self.fmp.get_balance_sheet(ticker),
                    "profile": self.fmp.get_profile(ticker),
                }
            except Exception as e:
                logger.warning(f"FMP fundamental fetch failed for {ticker}: {e}")

        # Fallback to Yahoo
        return self.yahoo.get_financials(ticker)

    def get_macro_data(self) -> dict:
        """Fetch macro indicators from FRED."""
        if self.fred:
            return self.fred.get_macro_indicators()
        return {}

    def status(self) -> dict:
        """Return pipeline status."""
        return self.config.status_report()
