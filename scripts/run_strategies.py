#!/usr/bin/env python3
"""
Strategy Runner — Runs all 4 Sprint 1 equity strategies with live data.
Fetches real market data from Yahoo Finance, computes signals, and
reports portfolio construction results.

This is the integration test that validates the full pipeline:
    Data Ingestion → Feature Generation → Signal → Position Sizing → Risk Check
"""

import sys
import os
import time
import warnings
warnings.filterwarnings("ignore")

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "strategies"))

import pandas as pd
import numpy as np
from datetime import date, datetime

from backend.app.services.data_ingestion.pipeline import DataIngestionPipeline

# Import strategies
from strategies.equity.momentum.cross_sectional import CrossSectionalMomentum
from strategies.equity.value.enhanced_value import EnhancedValueComposite
from strategies.equity.low_volatility.low_vol import LowVolatilityAnomaly
from strategies.equity.residual_momentum.residual_mom import ResidualMomentum
from strategies.base import StrategyContext


def fetch_universe_data(pipeline, tickers, period="2y"):
    """Fetch price and volume data for the universe."""
    print(f"  Fetching data for {len(tickers)} tickers ({period})...")
    t0 = time.time()

    raw = pipeline.yahoo.get_prices(tickers, period=period)
    elapsed = time.time() - t0
    print(f"  ✅ Data fetched in {elapsed:.1f}s, shape: {raw.shape}")

    # Parse multi-level columns
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw.xs("Close", axis=1, level=0) if "Close" in raw.columns.get_level_values(0) else raw.xs("Close", axis=1, level=1)
        volume = raw.xs("Volume", axis=1, level=0) if "Volume" in raw.columns.get_level_values(0) else raw.xs("Volume", axis=1, level=1)
    else:
        close = raw.filter(like="Close")
        volume = raw.filter(like="Volume")

    return close, volume


def run_strategy(strategy, close, volume, extra_data=None):
    """Run a single strategy through the full pipeline."""
    meta = strategy.get_metadata()
    print(f"\n{'='*60}")
    print(f"  Strategy: {meta.code} — {meta.name}")
    print(f"  Style: {meta.style.value} | Horizon: {meta.horizon}")
    print(f"  Directionality: {meta.directionality}")
    print(f"{'='*60}")

    # Build data dict
    data = {"close": close, "volume": volume}
    if extra_data:
        data.update(extra_data)

    # Build context
    context = StrategyContext(
        as_of_date=date.today(),
        universe=list(close.columns),
        parameters=strategy._params,
    )

    # 1. Generate features
    t0 = time.time()
    try:
        features = strategy.generate_features(context, data)
        feat_time = time.time() - t0
        print(f"  ✅ Features: {features.shape[0]} stocks × {features.shape[1]} features ({feat_time:.1f}s)")
    except Exception as e:
        print(f"  ❌ Feature generation failed: {e}")
        return None

    # 2. Generate signal
    t0 = time.time()
    try:
        signal = strategy.generate_signal(features, strategy._params)
        sig_time = time.time() - t0
        n_long = (signal > 0).sum()
        n_short = (signal < 0).sum()
        print(f"  ✅ Signal: {n_long} longs, {n_short} shorts ({sig_time:.2f}s)")
    except Exception as e:
        print(f"  ❌ Signal generation failed: {e}")
        return None

    # 3. Size positions
    t0 = time.time()
    risk_context = {
        "limits": {
            "max_gross": strategy._params.get("book_size", 1_000_000) * 2,
            "max_single_name": strategy._params.get("book_size", 1_000_000) * strategy._params.get("max_single_weight", 0.05),
        }
    }
    targets = strategy.size_positions(signal, risk_context, strategy._params)
    size_time = time.time() - t0

    gross = targets.abs().sum()
    net = targets.sum()
    n_positions = (targets != 0).sum()
    print(f"  ✅ Positions: {n_positions} names, Gross: ${gross:,.0f}, Net: ${net:,.0f} ({size_time:.2f}s)")

    # 4. Risk check
    risk_result = strategy.check_risk(targets, risk_context)
    if risk_result.passed:
        print(f"  ✅ Risk check: PASSED")
    else:
        print(f"  ⚠️  Risk check: FAILED")
        for breach in risk_result.hard_breaches:
            print(f"     Hard breach: {breach}")
    for warn in risk_result.soft_warnings:
        print(f"     Warning: {warn}")

    # 5. Build orders (from flat)
    current = pd.Series(0, index=targets.index)
    orders = strategy.build_orders(current, targets)
    n_buys = sum(1 for o in orders if o.side == "buy")
    n_sells = sum(1 for o in orders if o.side == "sell")
    print(f"  ✅ Orders: {len(orders)} total ({n_buys} buys, {n_sells} sells)")

    # Top holdings
    top_long = targets[targets > 0].sort_values(ascending=False).head(5)
    top_short = targets[targets < 0].sort_values().head(5)

    if len(top_long) > 0:
        print(f"\n  📈 Top 5 Longs:")
        for sym, val in top_long.items():
            print(f"     {sym:8s} ${val:>10,.0f}  ({val/strategy._params['book_size']*100:>5.1f}%)")

    if len(top_short) > 0:
        print(f"\n  📉 Top 5 Shorts:")
        for sym, val in top_short.items():
            print(f"     {sym:8s} ${val:>10,.0f}  ({val/strategy._params['book_size']*100:>5.1f}%)")

    return {
        "code": meta.code,
        "name": meta.name,
        "n_long": n_long,
        "n_short": n_short,
        "gross": gross,
        "net": net,
        "risk_passed": risk_result.passed,
        "n_orders": len(orders),
    }


def main():
    print("=" * 70)
    print("151 Trading System — Sprint 1 Strategy Runner")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Initialize pipeline
    pipeline = DataIngestionPipeline()

    # Get universe (top 50 S&P 500 by market cap for speed)
    print("\n📊 Loading equity universe...")
    universe = pipeline.get_equity_universe()[:50]
    print(f"  Universe: {len(universe)} tickers")

    # Fetch data
    print("\n📡 Fetching market data...")
    close, volume = fetch_universe_data(pipeline, universe, period="2y")

    # Drop tickers with too many NaN
    valid_tickers = close.columns[close.isna().sum() < len(close) * 0.3]
    close = close[valid_tickers]
    volume = volume[valid_tickers] if volume is not None else None
    print(f"  Valid tickers after filtering: {len(valid_tickers)}")

    # Fetch SPY as benchmark
    print("\n📡 Fetching benchmark (SPY)...")
    spy_data = pipeline.yahoo.get_prices(["SPY"], period="2y")
    if isinstance(spy_data.columns, pd.MultiIndex):
        benchmark = spy_data.xs("Close", axis=1, level=0) if "Close" in spy_data.columns.get_level_values(0) else spy_data.xs("Close", axis=1, level=1)
    else:
        benchmark = spy_data.filter(like="Close")
    print(f"  SPY data: {len(benchmark)} days")

    # Run all 4 strategies
    results = []

    # 1. Cross-Sectional Momentum
    mom = CrossSectionalMomentum()
    r = run_strategy(mom, close, volume)
    if r:
        results.append(r)

    # 2. Enhanced Value Composite (needs fundamental data)
    # Create synthetic fundamentals from available data for testing
    print("\n  📊 Fetching fundamental data from FMP...")
    fundamentals_data = {}
    if pipeline.fmp:
        sample_tickers = list(valid_tickers[:10])  # Sample for speed
        for ticker in sample_tickers:
            try:
                ratios = pipeline.fmp.get_financial_ratios(ticker, limit=2)
                metrics = pipeline.fmp.get_key_metrics(ticker, limit=2)
                profile = pipeline.fmp.get_profile(ticker)
                if ratios and metrics and profile:
                    latest_ratio = ratios[0]
                    latest_metric = metrics[0]
                    prof = profile[0]
                    fundamentals_data[ticker] = {
                        "bookValuePerShare": latest_metric.get("bookValuePerShare", 0),
                        "netIncomePerShare": latest_metric.get("netIncomePerShare", 0),
                        "operatingCashFlowPerShare": latest_metric.get("operatingCashFlowPerShare", 0),
                        "revenuePerShare": latest_metric.get("revenuePerShare", 0),
                        "price": prof.get("price", close[ticker].iloc[-1] if ticker in close.columns else 1),
                        "marketCap": prof.get("mktCap", 0),
                        "sector": prof.get("sector", "Unknown"),
                        "returnOnAssets": latest_ratio.get("returnOnAssets", 0),
                        "currentRatio": latest_ratio.get("currentRatio", 1),
                        "grossProfitMargin": latest_ratio.get("grossProfitMargin", 0),
                    }
            except Exception as e:
                pass

    if fundamentals_data:
        fund_df = pd.DataFrame(fundamentals_data).T
        print(f"  ✅ Fundamentals loaded for {len(fund_df)} tickers")
        val = EnhancedValueComposite({"min_market_cap": 0})  # Lower threshold for test
        r = run_strategy(val, close[fund_df.index.intersection(close.columns)], 
                        volume[fund_df.index.intersection(volume.columns)] if volume is not None else None,
                        {"fundamentals": fund_df})
        if r:
            results.append(r)
    else:
        print("  ⚠️  Skipping Value strategy (no fundamental data available)")

    # 3. Low Volatility Anomaly
    lvol = LowVolatilityAnomaly()
    r = run_strategy(lvol, close, volume, {"benchmark": benchmark})
    if r:
        results.append(r)

    # 4. Residual Momentum
    rmom = ResidualMomentum()
    r = run_strategy(rmom, close, volume)
    if r:
        results.append(r)

    # Summary
    print("\n" + "=" * 70)
    print("📋 STRATEGY RUNNER SUMMARY")
    print("=" * 70)
    if results:
        summary = pd.DataFrame(results)
        print(f"\n{'Code':<15} {'Name':<30} {'Long':>5} {'Short':>5} {'Gross':>12} {'Risk':>6}")
        print("-" * 75)
        for _, row in summary.iterrows():
            risk_icon = "✅" if row["risk_passed"] else "⚠️"
            print(f"{row['code']:<15} {row['name']:<30} {row['n_long']:>5} {row['n_short']:>5} ${row['gross']:>10,.0f} {risk_icon}")
        print("-" * 75)
        print(f"Total strategies run: {len(results)}")
        print(f"All risk checks passed: {all(r['risk_passed'] for r in results)}")
    else:
        print("  No strategies completed successfully.")

    print("=" * 70)
    return len(results) > 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
