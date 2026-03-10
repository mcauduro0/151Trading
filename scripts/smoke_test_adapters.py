"""Smoke test for data provider adapters.

Tests Yahoo Finance adapter (no API key needed) with real data fetch.
"""

import asyncio
import sys
sys.path.insert(0, "/home/ubuntu/151Trading/backend")
sys.path.insert(0, "/home/ubuntu/151Trading")

from app.adapters.data_providers.yahoo_finance.adapter import YahooFinanceAdapter


async def test_yahoo_finance():
    """Test Yahoo Finance adapter with real data."""
    print("=" * 60)
    print("SMOKE TEST: Yahoo Finance Adapter")
    print("=" * 60)

    adapter = YahooFinanceAdapter()

    # Test connection
    print("\n1. Testing connection...")
    connected = await adapter.connect()
    print(f"   Connection: {'OK' if connected else 'FAILED'}")

    # Test daily bars
    print("\n2. Fetching daily bars for SPY, AAPL, MSFT...")
    bars = await adapter.fetch_daily_bars(
        symbols=["SPY", "AAPL", "MSFT"],
        start_date="2025-01-01",
    )
    print(f"   Rows fetched: {len(bars)}")
    if not bars.empty:
        print(f"   Columns: {list(bars.columns)}")
        print(f"   Symbols: {bars['symbol'].unique().tolist()}")
        print(f"   Date range: {bars['ts'].min()} to {bars['ts'].max()}")
        print(f"   Sample data:")
        print(bars.groupby("symbol").tail(1)[["symbol", "ts", "close", "volume"]].to_string(index=False))

    # Test data validation
    print("\n3. Running data validation...")
    warnings = await adapter.validate_data(bars)
    if warnings:
        for w in warnings:
            print(f"   WARNING: {w}")
    else:
        print("   Validation: PASSED (no warnings)")

    # Test fundamentals
    print("\n4. Fetching fundamentals for AAPL...")
    fundamentals = await adapter.fetch_fundamentals(["AAPL"])
    print(f"   Facts fetched: {len(fundamentals)}")
    if not fundamentals.empty:
        print(f"   Available facts: {fundamentals['fact_name'].tolist()}")

    # Test health check
    print("\n5. Health check...")
    health = await adapter.health_check()
    print(f"   Status: {health['status']}")

    print("\n" + "=" * 60)
    print("SMOKE TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_yahoo_finance())
