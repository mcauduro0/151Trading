#!/usr/bin/env python3
"""
Live Data Ingestion Test — Verifies all configured API connections.
Tests each data provider with a small request and reports status.
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from app.services.data_ingestion.pipeline import DataIngestionPipeline

def test_pipeline():
    print("=" * 70)
    print("151 Trading System — Live Data Ingestion Test")
    print("=" * 70)

    pipeline = DataIngestionPipeline()

    # 1. Status Report
    print("\n📊 Provider Status:")
    status = pipeline.status()
    for provider, info in status.items():
        icon = "✅" if info["enabled"] else "❌"
        key_icon = "🔑" if info["has_key"] else "⚠️"
        print(f"  {icon} {provider:20s} | Key: {key_icon} | Enabled: {info['enabled']}")

    results = {}

    # 2. Yahoo Finance
    print("\n" + "-" * 50)
    print("🔸 Testing Yahoo Finance...")
    try:
        t0 = time.time()
        data = pipeline.yahoo.get_prices(["AAPL", "MSFT", "GOOGL"], period="5d")
        elapsed = time.time() - t0
        print(f"  ✅ Fetched 3 tickers, shape: {data.shape}, latency: {elapsed:.1f}s")
        results["yahoo"] = True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        results["yahoo"] = False

    # 3. FRED
    print("\n" + "-" * 50)
    print("🔸 Testing FRED...")
    if pipeline.fred:
        try:
            t0 = time.time()
            vix = pipeline.fred.get_series("VIXCLS")
            elapsed = time.time() - t0
            print(f"  ✅ VIX series: {len(vix)} observations, latest: {vix.iloc[-1]:.2f}, latency: {elapsed:.1f}s")
            results["fred"] = True
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results["fred"] = False
    else:
        print("  ⏭️  Skipped (not configured)")
        results["fred"] = None

    # 4. FMP
    print("\n" + "-" * 50)
    print("🔸 Testing FMP...")
    if pipeline.fmp:
        try:
            t0 = time.time()
            profile = pipeline.fmp.get_profile("AAPL")
            elapsed = time.time() - t0
            if profile:
                p = profile[0]
                print(f"  ✅ AAPL: {p.get('companyName', 'N/A')}, MktCap: ${p.get('mktCap', 0)/1e9:.0f}B, latency: {elapsed:.1f}s")
            results["fmp"] = True
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results["fmp"] = False

        # Test SP500 constituents
        try:
            t0 = time.time()
            sp500 = pipeline.fmp.get_sp500_constituents()
            elapsed = time.time() - t0
            print(f"  ✅ S&P 500 constituents: {len(sp500)} stocks, latency: {elapsed:.1f}s")
        except Exception as e:
            print(f"  ⚠️  SP500 constituents: {e}")
    else:
        print("  ⏭️  Skipped (not configured)")
        results["fmp"] = None

    # 5. Trading Economics
    print("\n" + "-" * 50)
    print("🔸 Testing Trading Economics...")
    if pipeline.te:
        try:
            t0 = time.time()
            markets = pipeline.te.get_markets("index")
            elapsed = time.time() - t0
            print(f"  ✅ Market indices: {len(markets)} entries, latency: {elapsed:.1f}s")
            if markets:
                print(f"     Sample: {markets[0].get('Name', 'N/A')} = {markets[0].get('Last', 'N/A')}")
            results["te"] = True
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results["te"] = False
    else:
        print("  ⏭️  Skipped (not configured)")
        results["te"] = None

    # 6. Equity Universe
    print("\n" + "-" * 50)
    print("🔸 Testing Equity Universe...")
    try:
        t0 = time.time()
        universe = pipeline.get_equity_universe()
        elapsed = time.time() - t0
        print(f"  ✅ Universe: {len(universe)} tickers, latency: {elapsed:.1f}s")
        print(f"     Sample: {universe[:10]}")
        results["universe"] = True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        results["universe"] = False

    # Summary
    print("\n" + "=" * 70)
    print("📋 SUMMARY")
    print("=" * 70)
    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    skipped = sum(1 for v in results.values() if v is None)
    print(f"  ✅ Passed: {passed} | ❌ Failed: {failed} | ⏭️  Skipped: {skipped}")
    print("=" * 70)

    return failed == 0


if __name__ == "__main__":
    success = test_pipeline()
    sys.exit(0 if success else 1)
