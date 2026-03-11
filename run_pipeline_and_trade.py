"""
GB Trading — Run Pipeline and Execute Orders on Alpaca Paper Trading
====================================================================
Runs the full daily pipeline, then submits the generated orders
to Alpaca Paper Trading API.

Only submits BUY orders for STOCKS (no ETFs, no funds).
"""

import os
import sys
import json
import time
import logging
import requests
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from strategies.automation.daily_pipeline import DailyPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("gb_trading.executor")

# ETFs and funds to exclude (user only wants stocks)
EXCLUDED_TICKERS = {
    "SPY", "QQQ", "IWM", "DIA",
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC",
    "TLT", "IEF", "SHY", "LQD", "HYG", "TIP", "AGG",
    "GLD", "SLV", "USO", "UNG", "DBA", "DBB",
    "UUP", "FXE", "FXY", "FXB", "FXA",
    "VXX", "SVXY", "UVXY",
    "BTC-USD", "ETH-USD", "SOL-USD",
}

ALPACA_BASE = "https://paper-api.alpaca.markets"
ALPACA_KEY = os.environ.get("ALPACA_API_KEY", "")
ALPACA_SECRET = os.environ.get("ALPACA_SECRET_KEY", "")

HEADERS = {
    "APCA-API-KEY-ID": ALPACA_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET,
    "Content-Type": "application/json",
}


def get_account():
    """Get Alpaca account info."""
    resp = requests.get(f"{ALPACA_BASE}/v2/account", headers=HEADERS, timeout=10)
    return resp.json()


def get_positions():
    """Get current positions."""
    resp = requests.get(f"{ALPACA_BASE}/v2/positions", headers=HEADERS, timeout=10)
    return resp.json()


def submit_order(symbol, qty, side, order_type="market", time_in_force="day"):
    """Submit an order to Alpaca."""
    payload = {
        "symbol": symbol,
        "qty": str(qty),
        "side": side,
        "type": order_type,
        "time_in_force": time_in_force,
    }
    resp = requests.post(
        f"{ALPACA_BASE}/v2/orders",
        headers=HEADERS,
        json=payload,
        timeout=10,
    )
    return resp.json(), resp.status_code


def main():
    logger.info("=" * 60)
    logger.info("GB Trading — Daily Pipeline + Alpaca Execution")
    logger.info("=" * 60)

    # Step 1: Run the pipeline
    logger.info("Running daily pipeline...")
    pipeline = DailyPipeline()
    report = pipeline.run_full_pipeline()
    report_dict = pipeline.get_report_dict()

    logger.info(f"Pipeline status: {report.overall_status}")
    logger.info(f"Total signals: {report.total_signals}")
    logger.info(f"Total orders generated: {report.total_orders}")

    if report.overall_status == "FAILED":
        logger.error("Pipeline FAILED. Not submitting orders.")
        print(json.dumps(report_dict, indent=2))
        return

    # Step 2: Get current account and positions
    account = get_account()
    buying_power = float(account.get("buying_power", 0))
    equity = float(account.get("equity", 0))
    logger.info(f"Account equity: ${equity:,.0f}, buying power: ${buying_power:,.0f}")

    positions = get_positions()
    held_symbols = {p["symbol"] for p in positions} if isinstance(positions, list) else set()
    logger.info(f"Currently holding {len(held_symbols)} positions: {sorted(held_symbols)}")

    # Step 3: Filter orders — STOCKS ONLY, no ETFs/funds
    order_stage = None
    for stage in report.stages:
        if stage.stage.value == "order_generation":
            order_stage = stage
            break

    if not order_stage or not order_stage.details.get("orders"):
        logger.info("No orders to execute.")
        return

    raw_orders = order_stage.details["orders"]
    
    # Filter: only stocks, only buy orders, not already held, deduplicate by ticker
    seen_tickers = set()
    filtered_orders = []
    for order in raw_orders:
        ticker = order["ticker"]
        if ticker in EXCLUDED_TICKERS:
            logger.info(f"  SKIP {ticker}: ETF/fund excluded")
            continue
        if ticker in held_symbols:
            logger.info(f"  SKIP {ticker}: already held")
            continue
        if order["side"] != "buy":
            logger.info(f"  SKIP {ticker}: sell order (keeping positions)")
            continue
        if ticker in seen_tickers:
            continue  # deduplicate
        seen_tickers.add(ticker)
        filtered_orders.append(order)

    logger.info(f"\nFiltered to {len(filtered_orders)} buy orders (stocks only, not held)")

    if not filtered_orders:
        logger.info("No new orders to submit after filtering.")
        return

    # Step 4: Size orders — equal weight, max 5% per position
    max_position_pct = 0.05
    max_per_position = equity * max_position_pct
    available_per_order = min(buying_power / max(len(filtered_orders), 1), max_per_position)

    if available_per_order < 100:
        logger.warning(f"Insufficient buying power (${buying_power:,.0f}). Cannot place orders.")
        return

    # Step 5: Submit orders
    submitted = []
    failed = []

    for order in filtered_orders:
        ticker = order["ticker"]
        price = order.get("signal_strength", 0)  # We'll use market orders
        qty = max(1, int(available_per_order / max(price * 100, 1)))  # Rough sizing
        
        # Use notional amount instead for more precise sizing
        notional = round(available_per_order, 2)
        
        logger.info(f"  Submitting: BUY {ticker} ~${notional:,.0f} (market)")
        
        try:
            # Use notional orders for fractional shares
            payload = {
                "symbol": ticker,
                "notional": str(notional),
                "side": "buy",
                "type": "market",
                "time_in_force": "day",
            }
            resp = requests.post(
                f"{ALPACA_BASE}/v2/orders",
                headers=HEADERS,
                json=payload,
                timeout=10,
            )
            result = resp.json()
            
            if resp.status_code in (200, 201):
                submitted.append({"symbol": ticker, "notional": notional, "order_id": result.get("id")})
                logger.info(f"    ✓ Order accepted: {result.get('id', 'unknown')}")
            else:
                failed.append({"symbol": ticker, "error": result.get("message", str(result))})
                logger.warning(f"    ✗ Order rejected: {result.get('message', str(result))}")
        except Exception as e:
            failed.append({"symbol": ticker, "error": str(e)})
            logger.error(f"    ✗ Order error: {e}")

        time.sleep(0.2)  # Rate limit

    # Step 6: Summary
    logger.info("\n" + "=" * 60)
    logger.info(f"  EXECUTION SUMMARY")
    logger.info(f"  Pipeline: {report.overall_status}")
    logger.info(f"  Signals generated: {report.total_signals}")
    logger.info(f"  Orders submitted: {len(submitted)}")
    logger.info(f"  Orders failed: {len(failed)}")
    logger.info("=" * 60)

    for s in submitted:
        logger.info(f"  ✓ {s['symbol']}: ${s['notional']:,.0f} (ID: {s['order_id']})")
    for f in failed:
        logger.info(f"  ✗ {f['symbol']}: {f['error']}")

    # Save report
    report_path = os.path.join(os.path.dirname(__file__), "pipeline_report.json")
    with open(report_path, "w") as fp:
        json.dump({
            "pipeline": report_dict,
            "execution": {
                "submitted": submitted,
                "failed": failed,
                "account_equity": equity,
                "buying_power": buying_power,
                "timestamp": datetime.utcnow().isoformat(),
            },
        }, fp, indent=2)
    logger.info(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
