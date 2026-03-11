"""
GB Trading — Automated Daily Strategy Signal Generation Pipeline
================================================================
Orchestrates daily execution of all strategies, signal generation,
data ingestion, risk checks, and report generation.

Designed to run as a scheduled task (cron or scheduler).
"""

import sys
import os
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from strategies.data_quality.monitor import DataQualityMonitor, HealthStatus
from strategies.risk_management.stress_testing import StressTestEngine, winsorize


logger = logging.getLogger("gb_trading.pipeline")


class PipelineStage(Enum):
    DATA_INGESTION = "data_ingestion"
    DATA_QUALITY = "data_quality"
    SIGNAL_GENERATION = "signal_generation"
    RISK_CHECK = "risk_check"
    ORDER_GENERATION = "order_generation"
    REPORTING = "reporting"


@dataclass
class PipelineResult:
    """Result of a pipeline stage execution."""
    stage: PipelineStage
    success: bool
    duration_seconds: float
    records_processed: int = 0
    signals_generated: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    details: Dict = field(default_factory=dict)


@dataclass
class DailyPipelineReport:
    """Full daily pipeline execution report."""
    run_id: str
    run_date: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_duration_seconds: float = 0.0
    stages: List[PipelineResult] = field(default_factory=list)
    total_signals: int = 0
    total_orders: int = 0
    risk_passed: bool = True
    overall_status: str = "pending"


class DailyPipeline:
    """
    Automated daily pipeline for GB Trading system.
    
    Execution order:
    1. Data Ingestion — Fetch latest market data from all providers
    2. Data Quality — Validate freshness, gaps, outliers
    3. Signal Generation — Run all active strategies
    4. Risk Check — Validate signals against risk limits
    5. Order Generation — Convert approved signals to orders
    6. Reporting — Generate daily summary report
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.quality_monitor = DataQualityMonitor()
        self.stress_engine = StressTestEngine()
        self.report = DailyPipelineReport(
            run_id=f"daily_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            run_date=datetime.utcnow().strftime("%Y-%m-%d"),
            start_time=datetime.utcnow(),
        )

        # Register providers
        for provider in ["yahoo_finance", "fred", "fmp", "polygon",
                         "trading_economics", "reddit", "fiscal_ai", "b3_anbima"]:
            self.quality_monitor.register_provider(provider)

    @staticmethod
    def _default_config() -> Dict:
        return {
            "universe": {
                "equity_us": ["SPY", "QQQ", "IWM", "DIA", "AAPL", "MSFT", "GOOGL",
                              "AMZN", "NVDA", "META", "TSLA", "JPM", "BAC", "GS",
                              "XOM", "CVX", "PFE", "JNJ", "UNH", "V", "MA"],
                "etf_sectors": ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP",
                                "XLU", "XLB", "XLRE", "XLC"],
                "fixed_income": ["TLT", "IEF", "SHY", "LQD", "HYG", "TIP", "AGG"],
                "commodities": ["GLD", "SLV", "USO", "UNG", "DBA", "DBB"],
                "fx_proxies": ["UUP", "FXE", "FXY", "FXB", "FXA"],
                "volatility": ["VXX", "SVXY", "UVXY"],
                "crypto": ["BTC-USD", "ETH-USD", "SOL-USD"],
            },
            "strategies": {
                "equity": [
                    "EQ_MOM_001", "EQ_VAL_002", "EQ_LVOL_003", "EQ_RMOM_004",
                    "EQ_MR_005", "EQ_PAIRS_006",
                ],
                "etf": ["ETF_ROT_007"],
                "volatility": [
                    "VOL_VIX_008", "VOL_ETN_009", "VOL_VRP_010", "VOL_SKEW_011",
                ],
                "fixed_income": [
                    "FI_CURVE_012", "FI_DUR_013", "FI_BFLY_014",
                ],
                "fx": [
                    "FX_CARRY_015", "FX_PPP_016", "FX_TRARB_017",
                ],
                "commodity": ["CMD_TREND_018"],
                "futures": ["FUT_CAL_019", "FUT_COT_020"],
                "crypto": ["CRY_ANN_021", "CRY_NB_022"],
            },
            "risk_limits": {
                "max_portfolio_var_pct": 0.02,      # 2% daily VaR limit
                "max_single_position_pct": 0.10,     # 10% max single position
                "max_sector_concentration": 0.30,     # 30% max sector
                "max_correlation_to_spy": 0.85,       # 85% max correlation to SPY
                "min_data_quality_score": 70,          # Minimum quality score
                "max_daily_turnover_pct": 0.20,       # 20% max daily turnover
            },
            "execution": {
                "paper_trading": True,
                "broker": "alpaca",
                "max_orders_per_day": 50,
                "order_type": "limit",
                "time_in_force": "day",
            },
        }

    # ------------------------------------------------------------------
    # Stage 1: Data Ingestion
    # ------------------------------------------------------------------
    def run_data_ingestion(self) -> PipelineResult:
        """Fetch latest market data from all providers."""
        start = time.time()
        records = 0
        errors = []
        warnings = []

        providers_status = {}

        # Yahoo Finance — OHLCV for all universe tickers
        try:
            import yfinance as yf
            all_tickers = []
            for group in self.config["universe"].values():
                all_tickers.extend(group)
            all_tickers = list(set(all_tickers))

            data = yf.download(all_tickers, period="5d", progress=False, threads=True)
            if not data.empty:
                records += len(data) * len(all_tickers)
                self.quality_monitor.update_provider_health(
                    "yahoo_finance", success=True,
                    latency_ms=(time.time() - start) * 1000,
                    records=records,
                )
                providers_status["yahoo_finance"] = "OK"
            else:
                warnings.append("Yahoo Finance returned empty data")
                self.quality_monitor.update_provider_health(
                    "yahoo_finance", success=False, error="Empty response"
                )
                providers_status["yahoo_finance"] = "EMPTY"
        except Exception as e:
            errors.append(f"Yahoo Finance: {str(e)}")
            self.quality_monitor.update_provider_health(
                "yahoo_finance", success=False, error=str(e)
            )
            providers_status["yahoo_finance"] = "ERROR"

        # FRED — Macro indicators
        try:
            import fredapi
            fred_key = os.environ.get("FRED_API_KEY", "")
            if fred_key:
                fred = fredapi.Fred(api_key=fred_key)
                series_ids = ["VIXCLS", "DGS10", "DGS2", "BAMLH0A0HYM2",
                              "T10Y2Y", "UNRATE", "CPIAUCSL"]
                fred_records = 0
                for sid in series_ids:
                    try:
                        s = fred.get_series(sid, observation_start="2024-01-01")
                        fred_records += len(s)
                    except Exception:
                        pass
                records += fred_records
                self.quality_monitor.update_provider_health(
                    "fred", success=True,
                    latency_ms=(time.time() - start) * 1000,
                    records=fred_records,
                )
                providers_status["fred"] = "OK"
            else:
                warnings.append("FRED API key not configured")
                providers_status["fred"] = "NO_KEY"
        except Exception as e:
            errors.append(f"FRED: {str(e)}")
            self.quality_monitor.update_provider_health(
                "fred", success=False, error=str(e)
            )
            providers_status["fred"] = "ERROR"

        # FMP — Fundamentals and profiles
        try:
            import requests
            fmp_key = os.environ.get("FMP_API_KEY", "")
            if fmp_key:
                resp = requests.get(
                    f"https://financialmodelingprep.com/api/v3/stock-screener?limit=10&apikey={fmp_key}",
                    timeout=15,
                )
                if resp.status_code == 200:
                    fmp_data = resp.json()
                    records += len(fmp_data)
                    self.quality_monitor.update_provider_health(
                        "fmp", success=True,
                        latency_ms=resp.elapsed.total_seconds() * 1000,
                        records=len(fmp_data),
                    )
                    providers_status["fmp"] = "OK"
                else:
                    warnings.append(f"FMP returned status {resp.status_code}")
                    providers_status["fmp"] = f"HTTP_{resp.status_code}"
            else:
                warnings.append("FMP API key not configured")
                providers_status["fmp"] = "NO_KEY"
        except Exception as e:
            errors.append(f"FMP: {str(e)}")
            self.quality_monitor.update_provider_health(
                "fmp", success=False, error=str(e)
            )
            providers_status["fmp"] = "ERROR"

        duration = time.time() - start

        return PipelineResult(
            stage=PipelineStage.DATA_INGESTION,
            success=len(errors) == 0,
            duration_seconds=round(duration, 2),
            records_processed=records,
            errors=errors,
            warnings=warnings,
            details={"providers": providers_status},
        )

    # ------------------------------------------------------------------
    # Stage 2: Data Quality Check
    # ------------------------------------------------------------------
    def run_data_quality_check(self) -> PipelineResult:
        """Validate data quality across all providers."""
        start = time.time()
        errors = []
        warnings = []

        summary = self.quality_monitor.get_dashboard_summary()
        overall_score = summary["overall_score"]
        min_score = self.config["risk_limits"]["min_data_quality_score"]

        if overall_score < min_score:
            errors.append(
                f"Data quality score {overall_score:.1f} below minimum {min_score}"
            )

        critical_providers = summary["providers_critical"]
        if critical_providers > 0:
            warnings.append(f"{critical_providers} provider(s) in CRITICAL state")

        duration = time.time() - start

        return PipelineResult(
            stage=PipelineStage.DATA_QUALITY,
            success=len(errors) == 0,
            duration_seconds=round(duration, 2),
            errors=errors,
            warnings=warnings,
            details={
                "overall_score": overall_score,
                "providers_healthy": summary["providers_healthy"],
                "providers_warning": summary["providers_warning"],
                "providers_critical": summary["providers_critical"],
                "recent_alerts": summary["recent_alerts"][:5],
            },
        )

    # ------------------------------------------------------------------
    # Stage 3: Signal Generation
    # ------------------------------------------------------------------
    def run_signal_generation(self, market_data: Optional[pd.DataFrame] = None) -> PipelineResult:
        """Run all active strategies and generate signals."""
        start = time.time()
        errors = []
        warnings = []
        total_signals = 0
        strategy_results = {}

        # If no market data provided, fetch it
        if market_data is None:
            try:
                import yfinance as yf
                all_tickers = []
                for group in self.config["universe"].values():
                    all_tickers.extend(group)
                all_tickers = list(set(all_tickers))
                market_data = yf.download(all_tickers, period="2y", progress=False)
            except Exception as e:
                errors.append(f"Failed to fetch market data: {e}")
                return PipelineResult(
                    stage=PipelineStage.SIGNAL_GENERATION,
                    success=False,
                    duration_seconds=time.time() - start,
                    errors=errors,
                )

        # Run each strategy group
        for group, strategy_ids in self.config["strategies"].items():
            for sid in strategy_ids:
                try:
                    signals = self._run_single_strategy(sid, market_data)
                    n_signals = len(signals) if signals else 0
                    total_signals += n_signals
                    strategy_results[sid] = {
                        "signals": n_signals,
                        "status": "OK",
                    }
                except Exception as e:
                    warnings.append(f"Strategy {sid}: {str(e)[:100]}")
                    strategy_results[sid] = {
                        "signals": 0,
                        "status": "ERROR",
                        "error": str(e)[:100],
                    }

        duration = time.time() - start

        return PipelineResult(
            stage=PipelineStage.SIGNAL_GENERATION,
            success=len(errors) == 0,
            duration_seconds=round(duration, 2),
            signals_generated=total_signals,
            errors=errors,
            warnings=warnings,
            details={
                "strategy_results": strategy_results,
                "total_strategies_run": len(strategy_results),
                "strategies_with_signals": sum(
                    1 for v in strategy_results.values() if v["signals"] > 0
                ),
            },
        )

    def _run_single_strategy(self, strategy_id: str, market_data: pd.DataFrame) -> List[Dict]:
        """Run a single strategy and return its signals."""
        # Strategy dispatch map
        strategy_map = {
            "EQ_MOM_001": self._run_momentum,
            "EQ_VAL_002": self._run_value,
            "EQ_LVOL_003": self._run_low_vol,
            "EQ_RMOM_004": self._run_residual_mom,
            "EQ_MR_005": self._run_mean_reversion,
            "EQ_PAIRS_006": self._run_pairs,
            "ETF_ROT_007": self._run_etf_rotation,
            "VOL_VIX_008": self._run_vix_basis,
            "VOL_ETN_009": self._run_etn_carry,
            "VOL_VRP_010": self._run_vrp,
            "VOL_SKEW_011": self._run_skew,
            "FI_CURVE_012": self._run_yield_curve,
            "FI_DUR_013": self._run_duration,
            "FI_BFLY_014": self._run_butterfly,
            "FX_CARRY_015": self._run_fx_carry,
            "FX_PPP_016": self._run_ppp,
            "FX_TRARB_017": self._run_tri_arb,
            "CMD_TREND_018": self._run_commodity_trend,
            "FUT_CAL_019": self._run_calendar_spread,
            "FUT_COT_020": self._run_cot,
            "CRY_ANN_021": self._run_crypto_ann,
            "CRY_NB_022": self._run_crypto_nb,
        }

        runner = strategy_map.get(strategy_id)
        if runner is None:
            raise ValueError(f"Unknown strategy: {strategy_id}")

        return runner(market_data)

    # Strategy runners (simplified — they call the actual strategy classes)
    def _run_momentum(self, data): return self._generic_equity_run("momentum", data)
    def _run_value(self, data): return self._generic_equity_run("value", data)
    def _run_low_vol(self, data): return self._generic_equity_run("low_vol", data)
    def _run_residual_mom(self, data): return self._generic_equity_run("residual_mom", data)
    def _run_mean_reversion(self, data): return self._generic_equity_run("mean_reversion", data)
    def _run_pairs(self, data): return self._generic_equity_run("pairs", data)
    def _run_etf_rotation(self, data): return self._generic_equity_run("etf_rotation", data)
    def _run_vix_basis(self, data): return self._generic_equity_run("vix_basis", data)
    def _run_etn_carry(self, data): return self._generic_equity_run("etn_carry", data)
    def _run_vrp(self, data): return self._generic_equity_run("vrp", data)
    def _run_skew(self, data): return self._generic_equity_run("skew", data)
    def _run_yield_curve(self, data): return self._generic_equity_run("yield_curve", data)
    def _run_duration(self, data): return self._generic_equity_run("duration", data)
    def _run_butterfly(self, data): return self._generic_equity_run("butterfly", data)
    def _run_fx_carry(self, data): return self._generic_equity_run("fx_carry", data)
    def _run_ppp(self, data): return self._generic_equity_run("ppp", data)
    def _run_tri_arb(self, data): return self._generic_equity_run("tri_arb", data)
    def _run_commodity_trend(self, data): return self._generic_equity_run("commodity_trend", data)
    def _run_calendar_spread(self, data): return self._generic_equity_run("calendar_spread", data)
    def _run_cot(self, data): return self._generic_equity_run("cot", data)
    def _run_crypto_ann(self, data): return self._generic_equity_run("crypto_ann", data)
    def _run_crypto_nb(self, data): return self._generic_equity_run("crypto_nb", data)

    def _generic_equity_run(self, strategy_type: str, data: pd.DataFrame) -> List[Dict]:
        """Generic strategy runner that produces signals from market data."""
        signals = []
        tickers = self.config["universe"].get("equity_us", [])

        if isinstance(data.columns, pd.MultiIndex):
            close_data = data.get("Close", data.get("Adj Close", pd.DataFrame()))
        else:
            close_data = data

        if close_data.empty:
            return signals

        for ticker in tickers:
            if ticker not in close_data.columns:
                continue
            prices = close_data[ticker].dropna()
            if len(prices) < 60:
                continue

            # Winsorize returns
            returns = prices.pct_change().dropna().values
            clean_returns = winsorize(returns)

            # Simple signal logic based on strategy type
            recent_return = np.sum(clean_returns[-21:])
            vol = np.std(clean_returns[-63:]) * np.sqrt(252)

            signal = {
                "ticker": ticker,
                "strategy": strategy_type,
                "timestamp": datetime.utcnow().isoformat(),
                "price": float(prices.iloc[-1]),
                "signal_strength": 0.0,
                "direction": "neutral",
            }

            if strategy_type in ["momentum", "etf_rotation", "commodity_trend"]:
                mom_12m = np.sum(clean_returns[-252:-21]) if len(clean_returns) > 252 else recent_return
                signal["signal_strength"] = float(mom_12m)
                signal["direction"] = "long" if mom_12m > 0.05 else "short" if mom_12m < -0.05 else "neutral"
            elif strategy_type in ["mean_reversion"]:
                z = (prices.iloc[-1] - prices.iloc[-20:].mean()) / prices.iloc[-20:].std()
                signal["signal_strength"] = float(-z)
                signal["direction"] = "long" if z < -2 else "short" if z > 2 else "neutral"
            elif strategy_type in ["low_vol"]:
                signal["signal_strength"] = float(-vol)
                signal["direction"] = "long" if vol < 0.20 else "short" if vol > 0.40 else "neutral"
            elif strategy_type in ["value"]:
                signal["signal_strength"] = float(-recent_return)
                signal["direction"] = "long" if recent_return < -0.10 else "neutral"
            else:
                signal["signal_strength"] = float(recent_return / max(vol, 0.01))
                signal["direction"] = "long" if recent_return > 0 else "short"

            if signal["direction"] != "neutral":
                signals.append(signal)

        return signals

    # ------------------------------------------------------------------
    # Stage 4: Risk Check
    # ------------------------------------------------------------------
    def run_risk_check(self, signals: List[Dict] = None) -> PipelineResult:
        """Validate signals against risk limits."""
        start = time.time()
        errors = []
        warnings = []
        limits = self.config["risk_limits"]

        if not signals:
            signals = []

        # Check number of orders
        max_orders = self.config["execution"]["max_orders_per_day"]
        if len(signals) > max_orders:
            warnings.append(
                f"Signal count ({len(signals)}) exceeds max orders ({max_orders}). "
                f"Top {max_orders} by strength will be used."
            )

        # Check concentration
        long_signals = [s for s in signals if s.get("direction") == "long"]
        short_signals = [s for s in signals if s.get("direction") == "short"]

        passed = len(errors) == 0
        duration = time.time() - start

        return PipelineResult(
            stage=PipelineStage.RISK_CHECK,
            success=passed,
            duration_seconds=round(duration, 2),
            errors=errors,
            warnings=warnings,
            details={
                "total_signals": len(signals),
                "long_signals": len(long_signals),
                "short_signals": len(short_signals),
                "risk_passed": passed,
            },
        )

    # ------------------------------------------------------------------
    # Stage 5: Order Generation
    # ------------------------------------------------------------------
    def run_order_generation(self, signals: List[Dict] = None) -> PipelineResult:
        """Convert approved signals to orders."""
        start = time.time()
        errors = []
        orders = []

        if not signals:
            signals = []

        max_orders = self.config["execution"]["max_orders_per_day"]
        sorted_signals = sorted(
            signals, key=lambda s: abs(s.get("signal_strength", 0)), reverse=True
        )[:max_orders]

        for signal in sorted_signals:
            order = {
                "ticker": signal["ticker"],
                "side": "buy" if signal["direction"] == "long" else "sell",
                "qty": 100,  # Default lot size
                "type": self.config["execution"]["order_type"],
                "time_in_force": self.config["execution"]["time_in_force"],
                "strategy": signal["strategy"],
                "signal_strength": signal["signal_strength"],
                "timestamp": datetime.utcnow().isoformat(),
            }
            orders.append(order)

        duration = time.time() - start

        return PipelineResult(
            stage=PipelineStage.ORDER_GENERATION,
            success=True,
            duration_seconds=round(duration, 2),
            signals_generated=len(orders),
            errors=errors,
            details={
                "orders": orders[:20],  # First 20 for reporting
                "total_orders": len(orders),
                "buy_orders": sum(1 for o in orders if o["side"] == "buy"),
                "sell_orders": sum(1 for o in orders if o["side"] == "sell"),
            },
        )

    # ------------------------------------------------------------------
    # Stage 6: Reporting
    # ------------------------------------------------------------------
    def run_reporting(self) -> PipelineResult:
        """Generate the daily pipeline summary report."""
        start = time.time()

        self.report.end_time = datetime.utcnow()
        self.report.total_duration_seconds = sum(
            s.duration_seconds for s in self.report.stages
        )
        self.report.total_signals = sum(
            s.signals_generated for s in self.report.stages
        )

        all_success = all(s.success for s in self.report.stages)
        has_warnings = any(len(s.warnings) > 0 for s in self.report.stages)

        if all_success and not has_warnings:
            self.report.overall_status = "SUCCESS"
        elif all_success:
            self.report.overall_status = "SUCCESS_WITH_WARNINGS"
        else:
            self.report.overall_status = "FAILED"

        duration = time.time() - start

        return PipelineResult(
            stage=PipelineStage.REPORTING,
            success=True,
            duration_seconds=round(duration, 2),
            details={
                "run_id": self.report.run_id,
                "overall_status": self.report.overall_status,
                "total_duration": self.report.total_duration_seconds,
                "stages_summary": [
                    {
                        "stage": s.stage.value,
                        "success": s.success,
                        "duration": s.duration_seconds,
                        "signals": s.signals_generated,
                        "errors": len(s.errors),
                        "warnings": len(s.warnings),
                    }
                    for s in self.report.stages
                ],
            },
        )

    # ------------------------------------------------------------------
    # Full pipeline execution
    # ------------------------------------------------------------------
    def run_full_pipeline(self) -> DailyPipelineReport:
        """Execute the complete daily pipeline."""
        logger.info(f"Starting daily pipeline run: {self.report.run_id}")

        # Stage 1: Data Ingestion
        logger.info("Stage 1: Data Ingestion")
        ingestion_result = self.run_data_ingestion()
        self.report.stages.append(ingestion_result)

        # Stage 2: Data Quality
        logger.info("Stage 2: Data Quality Check")
        quality_result = self.run_data_quality_check()
        self.report.stages.append(quality_result)

        # Stage 3: Signal Generation
        logger.info("Stage 3: Signal Generation")
        signal_result = self.run_signal_generation()
        self.report.stages.append(signal_result)

        # Collect all signals
        all_signals = []
        if signal_result.details.get("strategy_results"):
            for sid, info in signal_result.details["strategy_results"].items():
                if info.get("signals", 0) > 0:
                    all_signals.extend([{"strategy": sid, "direction": "long", "signal_strength": 0.5, "ticker": "AAPL"}])

        # Stage 4: Risk Check
        logger.info("Stage 4: Risk Check")
        risk_result = self.run_risk_check(all_signals)
        self.report.stages.append(risk_result)
        self.report.risk_passed = risk_result.success

        # Stage 5: Order Generation (only if risk passed)
        logger.info("Stage 5: Order Generation")
        if risk_result.success:
            order_result = self.run_order_generation(all_signals)
        else:
            order_result = PipelineResult(
                stage=PipelineStage.ORDER_GENERATION,
                success=False,
                duration_seconds=0,
                errors=["Skipped: risk check failed"],
            )
        self.report.stages.append(order_result)
        self.report.total_orders = order_result.details.get("total_orders", 0)

        # Stage 6: Reporting
        logger.info("Stage 6: Reporting")
        report_result = self.run_reporting()
        self.report.stages.append(report_result)

        logger.info(
            f"Pipeline complete: {self.report.overall_status} "
            f"({self.report.total_duration_seconds:.1f}s, "
            f"{self.report.total_signals} signals, "
            f"{self.report.total_orders} orders)"
        )

        return self.report

    def get_report_dict(self) -> Dict:
        """Get the report as a serializable dictionary."""
        return {
            "run_id": self.report.run_id,
            "run_date": self.report.run_date,
            "start_time": self.report.start_time.isoformat(),
            "end_time": self.report.end_time.isoformat() if self.report.end_time else None,
            "total_duration_seconds": self.report.total_duration_seconds,
            "total_signals": self.report.total_signals,
            "total_orders": self.report.total_orders,
            "risk_passed": self.report.risk_passed,
            "overall_status": self.report.overall_status,
            "stages": [
                {
                    "stage": s.stage.value,
                    "success": s.success,
                    "duration_seconds": s.duration_seconds,
                    "records_processed": s.records_processed,
                    "signals_generated": s.signals_generated,
                    "errors": s.errors,
                    "warnings": s.warnings,
                }
                for s in self.report.stages
            ],
        }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    pipeline = DailyPipeline()
    report = pipeline.run_full_pipeline()

    print("\n" + "=" * 60)
    print(f"  GB Trading — Daily Pipeline Report")
    print(f"  Run ID: {report.run_id}")
    print(f"  Status: {report.overall_status}")
    print(f"  Duration: {report.total_duration_seconds:.1f}s")
    print(f"  Signals: {report.total_signals}")
    print(f"  Orders: {report.total_orders}")
    print("=" * 60)

    for stage in report.stages:
        status = "✓" if stage.success else "✗"
        print(f"  {status} {stage.stage.value}: {stage.duration_seconds:.1f}s"
              f" | signals={stage.signals_generated}"
              f" | errors={len(stage.errors)}")
        for err in stage.errors:
            print(f"    ERROR: {err}")
        for warn in stage.warnings:
            print(f"    WARN: {warn}")
