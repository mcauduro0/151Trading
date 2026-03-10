"""
Sprint 2 Strategy Runner — Integration Test with Live Data

Tests all 3 Sprint 2 strategies:
    1. EQ_MR_005: Bollinger Band Mean Reversion
    2. EQ_PAIRS_006: Ornstein-Uhlenbeck Pairs Trading
    3. ETF_ROT_007: ETF Sector Rotation

Fetches live data from Yahoo Finance and runs each strategy end-to-end.
"""

import sys, os
import time
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, timedelta

from strategies.base import StrategyContext
from strategies.equity.mean_reversion.bollinger_mr import BollingerMeanReversion
from strategies.equity.pairs_trading.ou_pairs import OUPairsTrading
from strategies.etf.rotation.etf_rotation import ETFSectorRotation


def fetch_data(tickers: list, period: str = "2y") -> dict:
    """Fetch OHLCV data from Yahoo Finance."""
    print(f"  Fetching data for {len(tickers)} tickers ({period})...")
    data = yf.download(tickers, period=period, progress=False, auto_adjust=True)
    
    result = {}
    if isinstance(data.columns, pd.MultiIndex):
        result["close"] = data["Close"]
        result["volume"] = data["Volume"]
        result["high"] = data["High"]
        result["low"] = data["Low"]
    else:
        result["close"] = data[["Close"]].rename(columns={"Close": tickers[0]})
        result["volume"] = data[["Volume"]].rename(columns={"Volume": tickers[0]})
    
    return result


def test_bollinger_mean_reversion():
    """Test EQ_MR_005 with S&P 500 large-cap stocks."""
    print("\n" + "=" * 70)
    print("TEST 1: Bollinger Band Mean Reversion (EQ_MR_005)")
    print("=" * 70)
    
    # Use a diverse set of large-cap stocks
    tickers = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM",
        "V", "UNH", "JNJ", "WMT", "PG", "MA", "HD", "BAC", "XOM", "CVX",
        "PFE", "ABBV", "KO", "PEP", "COST", "MRK", "TMO", "AVGO", "LLY",
        "ORCL", "CSCO", "ACN"
    ]
    
    data = fetch_data(tickers)
    
    # Relax filters for testing
    strategy = BollingerMeanReversion({
        "min_adv": 1_000_000,
        "min_price": 5.0,
        "min_bbw": 0.01,
        "max_bbw": 0.50,
        "pct_b_long_threshold": 0.15,
        "pct_b_short_threshold": 0.85,
        "use_rsi_confirmation": False,
        "volume_confirmation": False,
        "max_positions": 20,
        "book_size": 1_000_000,
    })
    
    meta = strategy.get_metadata()
    print(f"\n  Strategy: {meta.name} ({meta.code})")
    print(f"  Style: {meta.style.value} | Horizon: {meta.horizon}")
    
    context = StrategyContext(
        as_of_date=date.today(),
        universe=tickers,
        parameters=strategy._params,
    )
    
    # Generate features
    t0 = time.time()
    features = strategy.generate_features(context, data)
    t_feat = time.time() - t0
    print(f"\n  Features computed: {len(features)} stocks in {t_feat:.2f}s")
    
    if len(features) > 0:
        print(f"  Feature columns: {list(features.columns)}")
        
        # Show extreme %B values
        if "pct_b" in features.columns:
            oversold = features[features["pct_b"] < 0.15].sort_values("pct_b")
            overbought = features[features["pct_b"] > 0.85].sort_values("pct_b", ascending=False)
            print(f"\n  Oversold stocks (%B < 0.15): {len(oversold)}")
            for idx, row in oversold.head(5).iterrows():
                print(f"    {idx}: %B={row['pct_b']:.3f}, RSI={row['rsi_14']:.1f}, z={row['deviation_z']:.2f}")
            print(f"  Overbought stocks (%B > 0.85): {len(overbought)}")
            for idx, row in overbought.head(5).iterrows():
                print(f"    {idx}: %B={row['pct_b']:.3f}, RSI={row['rsi_14']:.1f}, z={row['deviation_z']:.2f}")
    
    # Generate signal
    t0 = time.time()
    signal = strategy.generate_signal(features, strategy._params)
    t_sig = time.time() - t0
    
    active = signal[signal != 0]
    longs = signal[signal > 0]
    shorts = signal[signal < 0]
    
    print(f"\n  Signal generated in {t_sig:.4f}s")
    print(f"  Active signals: {len(active)} (Longs: {len(longs)}, Shorts: {len(shorts)})")
    
    if len(longs) > 0:
        print(f"  Top longs: {dict(longs.sort_values(ascending=False).head(5).round(4))}")
    if len(shorts) > 0:
        print(f"  Top shorts: {dict(shorts.sort_values().head(5).round(4))}")
    
    # Position sizing
    targets = strategy.size_positions(signal, {}, strategy._params)
    active_targets = targets[targets != 0]
    print(f"\n  Position targets: {len(active_targets)} positions")
    print(f"  Gross exposure: ${targets.abs().sum():,.0f}")
    print(f"  Net exposure: ${targets.sum():,.0f}")
    
    # Risk check
    risk_result = strategy.check_risk(targets, {"limits": {"max_gross": 2_000_000}, "vix_level": 20})
    print(f"\n  Risk check: {'PASSED' if risk_result.passed else 'FAILED'}")
    for w in risk_result.soft_warnings:
        print(f"    ⚠️  {w}")
    for b in risk_result.hard_breaches:
        print(f"    ❌ {b}")
    
    print(f"\n  ✅ EQ_MR_005 Bollinger Mean Reversion: COMPLETE")
    return True


def test_ou_pairs_trading():
    """Test EQ_PAIRS_006 with tech sector pairs."""
    print("\n" + "=" * 70)
    print("TEST 2: Ornstein-Uhlenbeck Pairs Trading (EQ_PAIRS_006)")
    print("=" * 70)
    
    # Use same-sector stocks for better cointegration
    tickers = [
        # Tech
        "AAPL", "MSFT", "GOOGL", "META", "NVDA", "AVGO", "CSCO", "ORCL",
        # Financials
        "JPM", "BAC", "GS", "MS", "C", "WFC",
        # Energy
        "XOM", "CVX", "COP", "SLB",
    ]
    
    data = fetch_data(tickers)
    
    # Assign sectors
    sectors = pd.Series({
        "AAPL": "Tech", "MSFT": "Tech", "GOOGL": "Tech", "META": "Tech",
        "NVDA": "Tech", "AVGO": "Tech", "CSCO": "Tech", "ORCL": "Tech",
        "JPM": "Finance", "BAC": "Finance", "GS": "Finance", "MS": "Finance",
        "C": "Finance", "WFC": "Finance",
        "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "SLB": "Energy",
    })
    data["sectors"] = sectors
    
    strategy = OUPairsTrading({
        "lookback_days": 252,
        "min_correlation": 0.60,
        "same_sector_only": True,
        "coint_pvalue": 0.10,
        "max_half_life": 40,
        "entry_z": 1.5,
        "max_pairs": 10,
    })
    
    meta = strategy.get_metadata()
    print(f"\n  Strategy: {meta.name} ({meta.code})")
    print(f"  Style: {meta.style.value} | Complexity: {meta.complexity}")
    
    context = StrategyContext(
        as_of_date=date.today(),
        universe=tickers,
        parameters=strategy._params,
    )
    
    # Generate features (pair identification)
    t0 = time.time()
    features = strategy.generate_features(context, data)
    t_feat = time.time() - t0
    
    print(f"\n  Cointegrated pairs found: {len(features)} in {t_feat:.2f}s")
    
    if len(features) > 0:
        print(f"\n  Top pairs by cointegration strength:")
        for idx, row in features.head(8).iterrows():
            print(f"    {row['ticker_a']:>5} / {row['ticker_b']:<5} | "
                  f"corr={row['correlation']:.3f} | "
                  f"p={row['coint_pvalue']:.3f} | "
                  f"HL={row['half_life']:.1f}d | "
                  f"z={row['z_score']:.2f} | "
                  f"β={row['hedge_ratio']:.3f}")
    
    # Generate signal
    t0 = time.time()
    signal = strategy.generate_signal(features, strategy._params)
    t_sig = time.time() - t0
    
    active = signal[signal != 0]
    longs = signal[signal > 0]
    shorts = signal[signal < 0]
    
    print(f"\n  Signal generated in {t_sig:.4f}s")
    print(f"  Active positions: {len(active)} (Longs: {len(longs)}, Shorts: {len(shorts)})")
    
    if len(active) > 0:
        for sym, val in active.sort_values(ascending=False).items():
            direction = "LONG" if val > 0 else "SHORT"
            print(f"    {sym:>5}: {direction} ({val:+.4f})")
    
    # Position sizing
    targets = strategy.size_positions(signal, {}, strategy._params)
    active_targets = targets[targets != 0]
    print(f"\n  Position targets: {len(active_targets)} legs")
    print(f"  Gross exposure: ${targets.abs().sum():,.0f}")
    print(f"  Net exposure: ${targets.sum():,.0f} (target: ~$0 market neutral)")
    
    print(f"\n  ✅ EQ_PAIRS_006 OU Pairs Trading: COMPLETE")
    return True


def test_etf_rotation():
    """Test ETF_ROT_007 with sector ETFs."""
    print("\n" + "=" * 70)
    print("TEST 3: ETF Sector Rotation (ETF_ROT_007)")
    print("=" * 70)
    
    etf_tickers = ["XLK", "XLF", "XLV", "XLY", "XLP", "XLE", "XLI", "XLB", "XLRE", "XLU", "XLC", "SPY"]
    
    data = fetch_data(etf_tickers)
    
    # Separate benchmark
    if "SPY" in data["close"].columns:
        data["benchmark"] = data["close"]["SPY"]
    
    strategy = ETFSectorRotation({
        "n_long": 3,
        "n_short": 2,
        "use_macro_overlay": True,
        "book_size": 1_000_000,
    })
    
    meta = strategy.get_metadata()
    print(f"\n  Strategy: {meta.name} ({meta.code})")
    print(f"  Style: {meta.style.value} | Rebalance: {meta.horizon}")
    
    context = StrategyContext(
        as_of_date=date.today(),
        universe=etf_tickers,
        parameters=strategy._params,
    )
    
    # Generate features
    t0 = time.time()
    features = strategy.generate_features(context, data)
    t_feat = time.time() - t0
    
    print(f"\n  Sector features computed: {len(features)} ETFs in {t_feat:.2f}s")
    print(f"  Macro regime: {features['regime'].iloc[0]}")
    
    print(f"\n  Sector Rankings (by regime-adjusted momentum):")
    ranked = features.sort_values("regime_adjusted_score", ascending=False)
    for idx, row in ranked.iterrows():
        bar = "█" * int(abs(row["regime_adjusted_score"]) * 50)
        direction = "+" if row["regime_adjusted_score"] > 0 else "-"
        print(f"    {idx:>4} ({row['sector_name']:<25}) | "
              f"1m={row['ret_1m']:+.2%} | 3m={row['ret_3m']:+.2%} | 6m={row['ret_6m']:+.2%} | "
              f"score={row['regime_adjusted_score']:+.4f} | {row['sector_type']}")
    
    # Generate signal
    t0 = time.time()
    signal = strategy.generate_signal(features, strategy._params)
    t_sig = time.time() - t0
    
    longs = signal[signal > 0]
    shorts = signal[signal < 0]
    
    print(f"\n  Signal generated in {t_sig:.4f}s")
    print(f"  LONG sectors ({len(longs)}):")
    for sym, val in longs.sort_values(ascending=False).items():
        name = ETFSectorRotation.SECTOR_ETFS.get(sym, {}).get("name", "?")
        print(f"    {sym} ({name}): {val:+.4f}")
    print(f"  SHORT sectors ({len(shorts)}):")
    for sym, val in shorts.sort_values().items():
        name = ETFSectorRotation.SECTOR_ETFS.get(sym, {}).get("name", "?")
        print(f"    {sym} ({name}): {val:+.4f}")
    
    # Position sizing
    targets = strategy.size_positions(signal, {}, strategy._params)
    active_targets = targets[targets != 0]
    print(f"\n  Position targets:")
    for sym, val in active_targets.sort_values(ascending=False).items():
        name = ETFSectorRotation.SECTOR_ETFS.get(sym, {}).get("name", "?")
        print(f"    {sym} ({name}): ${val:+,.0f}")
    print(f"  Gross exposure: ${targets.abs().sum():,.0f}")
    print(f"  Net exposure: ${targets.sum():,.0f}")
    
    # Risk check
    risk_result = strategy.check_risk(targets, {"limits": {"max_gross": 2_000_000}, "vix_level": 20})
    print(f"\n  Risk check: {'PASSED' if risk_result.passed else 'FAILED'}")
    for w in risk_result.soft_warnings:
        print(f"    ⚠️  {w}")
    
    print(f"\n  ✅ ETF_ROT_007 ETF Sector Rotation: COMPLETE")
    return True


if __name__ == "__main__":
    print("=" * 70)
    print("  151 Trading System — Sprint 2 Strategy Integration Test")
    print(f"  Date: {date.today()} | Live Data from Yahoo Finance")
    print("=" * 70)
    
    results = {}
    
    try:
        results["EQ_MR_005"] = test_bollinger_mean_reversion()
    except Exception as e:
        print(f"\n  ❌ EQ_MR_005 FAILED: {e}")
        import traceback; traceback.print_exc()
        results["EQ_MR_005"] = False
    
    try:
        results["EQ_PAIRS_006"] = test_ou_pairs_trading()
    except Exception as e:
        print(f"\n  ❌ EQ_PAIRS_006 FAILED: {e}")
        import traceback; traceback.print_exc()
        results["EQ_PAIRS_006"] = False
    
    try:
        results["ETF_ROT_007"] = test_etf_rotation()
    except Exception as e:
        print(f"\n  ❌ ETF_ROT_007 FAILED: {e}")
        import traceback; traceback.print_exc()
        results["ETF_ROT_007"] = False
    
    # Summary
    print("\n" + "=" * 70)
    print("  SPRINT 2 INTEGRATION TEST SUMMARY")
    print("=" * 70)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    for code, ok in results.items():
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {code}: {status}")
    print(f"\n  Total: {passed}/{total} strategies passed")
    print("=" * 70)
