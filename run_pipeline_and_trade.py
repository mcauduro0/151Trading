"""
GB Trading — Daily Portfolio Manager + Signal Executor
=======================================================
Runs the full daily pipeline, then manages the Alpaca Paper Trading portfolio:

1. Generate signals from all 22 strategies
2. Evaluate current positions vs target allocation
3. EXIT positions that have SELL signals or lost momentum
4. ENTER new positions with BUY signals
5. REBALANCE existing positions toward equal weight (5% each)
6. Report execution summary

This script runs daily via GitHub Actions at 14:00 UTC (before US market open).
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

# ── Configuration ──────────────────────────────────────────────────────────
TARGET_WEIGHT = 0.055          # 5.5% per position (target for new entries)
MAX_WEIGHT = 0.15              # 15% hard cap before trimming
MIN_WEIGHT = 0.03              # 3% floor before topping up
REBALANCE_THRESHOLD = 0.015    # 1.5% drift triggers rebalance
MAX_POSITIONS = 20             # Maximum concurrent positions
MIN_ORDER_VALUE = 200          # Minimum order notional ($)
CASH_RESERVE_PCT = 0.10        # Keep 10% cash reserve

# ETFs, funds, and crypto to exclude (stocks only)
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


# ── Alpaca API Helpers ─────────────────────────────────────────────────────

def alpaca_get(endpoint):
    """GET request to Alpaca API."""
    resp = requests.get(f"{ALPACA_BASE}{endpoint}", headers=HEADERS, timeout=15)
    resp.raise_for_status()
    return resp.json()


def alpaca_post(endpoint, payload):
    """POST request to Alpaca API."""
    resp = requests.post(
        f"{ALPACA_BASE}{endpoint}",
        headers=HEADERS,
        json=payload,
        timeout=15,
    )
    return resp.json(), resp.status_code


def alpaca_delete(endpoint):
    """DELETE request to Alpaca API."""
    resp = requests.delete(f"{ALPACA_BASE}{endpoint}", headers=HEADERS, timeout=15)
    return resp.status_code


def get_account():
    return alpaca_get("/v2/account")


def get_positions():
    return alpaca_get("/v2/positions")


def cancel_all_orders():
    """Cancel all open orders."""
    return alpaca_delete("/v2/orders")


def submit_market_order(symbol, notional, side):
    """Submit a market order using notional amount (fractional shares)."""
    payload = {
        "symbol": symbol,
        "notional": str(round(notional, 2)),
        "side": side,
        "type": "market",
        "time_in_force": "day",
    }
    return alpaca_post("/v2/orders", payload)


def close_position(symbol):
    """Close an entire position."""
    return alpaca_delete(f"/v2/positions/{symbol}")


# ── Signal Processing ──────────────────────────────────────────────────────

def extract_signals(report):
    """Extract raw signals from pipeline report stages."""
    for stage in report.stages:
        if stage.stage.value == "signal_generation":
            return stage.details.get("raw_signals", [])
    return []


def build_signal_map(raw_signals):
    """
    Build a map of ticker -> aggregated signal.
    Combines signals from multiple strategies per ticker.
    Returns: {ticker: {"direction": "long"|"short", "strength": float, "strategies": [str]}}
    """
    from collections import defaultdict
    ticker_signals = defaultdict(lambda: {"long_strength": 0, "short_strength": 0, "strategies": []})

    for sig in raw_signals:
        ticker = sig.get("ticker", "")
        if ticker in EXCLUDED_TICKERS or not ticker:
            continue
        direction = sig.get("direction", "")
        strength = abs(sig.get("signal_strength", 0))
        strategy = sig.get("strategy", "unknown")

        if direction == "long":
            ticker_signals[ticker]["long_strength"] += strength
            ticker_signals[ticker]["strategies"].append(f"{strategy}:long")
        elif direction == "short":
            ticker_signals[ticker]["short_strength"] += strength
            ticker_signals[ticker]["strategies"].append(f"{strategy}:short")

    # Determine net direction per ticker
    signal_map = {}
    for ticker, data in ticker_signals.items():
        net = data["long_strength"] - data["short_strength"]
        if abs(net) > 0.01:  # Minimum signal threshold
            signal_map[ticker] = {
                "direction": "long" if net > 0 else "short",
                "strength": abs(net),
                "strategies": data["strategies"],
            }

    return signal_map


# ── Portfolio Management ───────────────────────────────────────────────────

def analyze_portfolio(positions, equity, signal_map, cash):
    """
    Analyze current portfolio and determine actions needed.
    
    Strategy:
    1. EXIT positions with ANY sell signal or stop-loss triggered
    2. REBALANCE overweight/underweight positions
    3. ENTER new tickers with buy signals (not currently held)
    4. REINFORCE existing positions with strong buy signals using excess cash
    
    Returns: exits, entries, rebalances
    """
    exits = []       # Positions to close
    entries = []     # New positions to open
    rebalances = []  # Existing positions to adjust

    held = {}
    for p in positions:
        sym = p["symbol"]
        mv = float(p["market_value"])
        weight = mv / equity if equity > 0 else 0
        pnl_pct = float(p["unrealized_plpc"])
        held[sym] = {
            "market_value": mv,
            "weight": weight,
            "pnl_pct": pnl_pct,
            "qty": float(p["qty"]),
        }

    # 1. EXITS: Close positions with SELL signals (any strength) or stop-loss
    for sym, pos_data in held.items():
        sig = signal_map.get(sym)
        should_exit = False
        reason = ""

        # Exit on ANY sell signal (not just strong ones)
        if sig and sig["direction"] == "short":
            should_exit = True
            reason = f"Sell signal (strength={sig['strength']:.2f})"
        elif pos_data["pnl_pct"] < -0.10:
            should_exit = True
            reason = f"Stop loss triggered (P&L={pos_data['pnl_pct']*100:.1f}%)"
        # No signal at all — keep position (neutral)

        if should_exit:
            exits.append({"symbol": sym, "reason": reason, "market_value": pos_data["market_value"]})

    # 2. REBALANCES: Trim overweight positions
    exit_symbols = set(e["symbol"] for e in exits)
    for sym, pos_data in held.items():
        if sym in exit_symbols:
            continue
        if pos_data["weight"] > MAX_WEIGHT:
            trim_amount = (pos_data["weight"] - TARGET_WEIGHT) * equity
            if trim_amount > MIN_ORDER_VALUE:
                rebalances.append({
                    "symbol": sym,
                    "action": "trim",
                    "amount": trim_amount,
                    "reason": f"Overweight ({pos_data['weight']*100:.1f}% → {TARGET_WEIGHT*100:.1f}%)",
                })

    # 3. ENTRIES: New tickers with BUY signals not currently held
    still_held = set(held.keys()) - exit_symbols
    buy_candidates = []
    for ticker, sig in signal_map.items():
        if sig["direction"] == "long" and ticker not in still_held:
            buy_candidates.append((ticker, sig["strength"], sig["strategies"]))

    buy_candidates.sort(key=lambda x: x[1], reverse=True)
    current_count = len(still_held)
    available_slots = MAX_POSITIONS - current_count

    for ticker, strength, strategies in buy_candidates[:available_slots]:
        if strength > 0.1:
            entries.append({
                "symbol": ticker,
                "strength": strength,
                "strategies": strategies,
            })

    # 4. REINFORCE: Deploy excess cash into positions with BUY signals
    #    Account for cash freed by exits and cash consumed by entries
    #    Target: keep invested at (1 - CASH_RESERVE_PCT) of equity
    exit_cash = sum(e["market_value"] for e in exits)  # Cash freed by exits
    entry_count = len(entries)  # Entries will consume cash
    
    # Estimate invested after exits + entries
    invested_after = sum(d["market_value"] for s, d in held.items() if s not in exit_symbols)
    # Entries will add to invested (estimated at target_weight * equity each)
    entry_investment = entry_count * TARGET_WEIGHT * equity
    invested_after += entry_investment
    
    target_invested = (1.0 - CASH_RESERVE_PCT) * equity  # 90%
    cash_to_deploy = target_invested - invested_after
    
    if cash_to_deploy > MIN_ORDER_VALUE:
        # Find candidates: held positions with BUY signals (not being exited)
        reinforce_candidates = []
        for sym in still_held:
            sig = signal_map.get(sym)
            if sig and sig["direction"] == "long":
                reinforce_candidates.append((sym, sig["strength"], held[sym]["market_value"]))
        
        reinforce_candidates.sort(key=lambda x: x[1], reverse=True)
        
        if reinforce_candidates:
            # Equal-weight distribution, capped at MAX_WEIGHT per position
            equal_share = cash_to_deploy / len(reinforce_candidates)
            
            for sym, strength, current_mv in reinforce_candidates:
                max_additional = (MAX_WEIGHT * equity) - current_mv
                share = min(equal_share, max(0, max_additional))
                if share > MIN_ORDER_VALUE:
                    rebalances.append({
                        "symbol": sym,
                        "action": "topup",
                        "amount": round(share, 2),
                        "reason": f"Deploy cash (signal={strength:.2f}, equal-weight, cap {MAX_WEIGHT*100:.0f}%)",
                    })

    return exits, entries, rebalances


# ── Main Execution ─────────────────────────────────────────────────────────

def main():
    logger.info("=" * 60)
    logger.info("  GB Trading — Daily Portfolio Manager")
    logger.info(f"  Date: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    logger.info("=" * 60)

    # ── Step 1: Run the pipeline ──
    logger.info("\n[STEP 1] Running daily signal pipeline...")
    pipeline = DailyPipeline()
    report = pipeline.run_full_pipeline()
    report_dict = pipeline.get_report_dict()

    logger.info(f"  Pipeline status: {report.overall_status}")
    logger.info(f"  Total signals: {report.total_signals}")

    # Extract and aggregate signals
    raw_signals = extract_signals(report)
    signal_map = build_signal_map(raw_signals)
    logger.info(f"  Unique tickers with signals: {len(signal_map)}")

    buy_signals = {k: v for k, v in signal_map.items() if v["direction"] == "long"}
    sell_signals = {k: v for k, v in signal_map.items() if v["direction"] == "short"}
    logger.info(f"  BUY signals: {len(buy_signals)} tickers")
    logger.info(f"  SELL signals: {len(sell_signals)} tickers")

    if report.overall_status == "FAILED" and report.total_signals == 0:
        logger.error("Pipeline FAILED with 0 signals. Aborting execution.")
        logger.error("Check pipeline errors:")
        for stage in report.stages:
            if stage.errors:
                logger.error(f"  {stage.stage.value}: {stage.errors}")
        print(json.dumps(report_dict, indent=2, default=str))
        sys.exit(1)

    # ── Step 2: Get current portfolio state ──
    logger.info("\n[STEP 2] Fetching current portfolio state...")
    account = get_account()
    equity = float(account.get("equity", 0))
    buying_power = float(account.get("buying_power", 0))
    cash = float(account.get("cash", 0))
    logger.info(f"  Equity: ${equity:,.0f}")
    logger.info(f"  Cash: ${cash:,.0f}")
    logger.info(f"  Buying power: ${buying_power:,.0f}")

    positions = get_positions()
    if not isinstance(positions, list):
        positions = []
    logger.info(f"  Current positions: {len(positions)}")
    for p in sorted(positions, key=lambda x: float(x["market_value"]), reverse=True):
        w = float(p["market_value"]) / equity * 100 if equity > 0 else 0
        pnl = float(p["unrealized_plpc"]) * 100
        logger.info(f"    {p['symbol']:6s}: ${float(p['market_value']):>10,.0f} ({w:.1f}%)  P&L: {pnl:>+.2f}%")

    # ── Step 3: Cancel any open orders ──
    logger.info("\n[STEP 3] Cancelling open orders...")
    cancel_all_orders()
    time.sleep(1)

    # ── Step 4: Analyze portfolio and determine actions ──
    logger.info("\n[STEP 4] Analyzing portfolio vs signals...")
    exits, entries, rebalances = analyze_portfolio(positions, equity, signal_map, cash)

    logger.info(f"  Exits planned: {len(exits)}")
    for e in exits:
        logger.info(f"    EXIT {e['symbol']}: {e['reason']}")

    logger.info(f"  Entries planned: {len(entries)}")
    for e in entries:
        logger.info(f"    ENTER {e['symbol']}: strength={e['strength']:.3f} ({', '.join(e['strategies'][:3])})")

    logger.info(f"  Rebalances planned: {len(rebalances)}")
    for r in rebalances:
        logger.info(f"    {r['action'].upper()} {r['symbol']}: ${r['amount']:,.0f} ({r['reason']})")

    # ── Step 5: Execute exits ──
    logger.info("\n[STEP 5] Executing exits...")
    exit_results = []
    for e in exits:
        sym = e["symbol"]
        logger.info(f"  Closing {sym}...")
        try:
            status = close_position(sym)
            if status in (200, 204):
                exit_results.append({"symbol": sym, "status": "closed", "reason": e["reason"]})
                logger.info(f"    ✓ {sym} closed")
            else:
                exit_results.append({"symbol": sym, "status": "failed", "reason": f"HTTP {status}"})
                logger.warning(f"    ✗ {sym} close failed: HTTP {status}")
        except Exception as ex:
            exit_results.append({"symbol": sym, "status": "error", "reason": str(ex)})
            logger.error(f"    ✗ {sym} error: {ex}")
        time.sleep(0.3)

    # Wait for exits to settle
    if exits:
        time.sleep(2)

    # ── Step 6: Execute rebalances ──
    logger.info("\n[STEP 6] Executing rebalances...")
    rebal_results = []
    for r in rebalances:
        sym = r["symbol"]
        action = r["action"]
        amount = r["amount"]

        side = "sell" if action == "trim" else "buy"
        logger.info(f"  {action.upper()} {sym}: {side} ${amount:,.0f}...")

        try:
            result, status = submit_market_order(sym, amount, side)
            if status in (200, 201):
                rebal_results.append({"symbol": sym, "action": action, "amount": amount, "status": "ok"})
                logger.info(f"    ✓ {sym} {action} ${amount:,.0f}")
            else:
                msg = result.get("message", str(result))
                rebal_results.append({"symbol": sym, "action": action, "status": "failed", "error": msg})
                logger.warning(f"    ✗ {sym} {action} failed: {msg}")
        except Exception as ex:
            rebal_results.append({"symbol": sym, "action": action, "status": "error", "error": str(ex)})
            logger.error(f"    ✗ {sym} error: {ex}")
        time.sleep(0.3)

    # ── Step 7: Execute entries ──
    logger.info("\n[STEP 7] Executing new entries...")

    # Refresh account after exits and rebalances
    account = get_account()
    equity = float(account.get("equity", 0))
    buying_power = float(account.get("buying_power", 0))
    cash_reserve = equity * CASH_RESERVE_PCT
    available_for_entries = max(0, buying_power - cash_reserve)

    entry_size = equity * TARGET_WEIGHT
    max_entries = int(available_for_entries / entry_size) if entry_size > 0 else 0
    actual_entries = entries[:max_entries]

    logger.info(f"  Available for entries: ${available_for_entries:,.0f}")
    logger.info(f"  Entry size: ${entry_size:,.0f} per position")
    logger.info(f"  Max new entries: {max_entries}")

    entry_results = []
    for e in actual_entries:
        sym = e["symbol"]
        notional = min(entry_size, available_for_entries)
        if notional < MIN_ORDER_VALUE:
            logger.info(f"  SKIP {sym}: insufficient buying power (${notional:,.0f})")
            continue

        logger.info(f"  BUY {sym}: ${notional:,.0f} (strength={e['strength']:.3f})...")
        try:
            result, status = submit_market_order(sym, notional, "buy")
            if status in (200, 201):
                entry_results.append({
                    "symbol": sym,
                    "notional": notional,
                    "order_id": result.get("id"),
                    "strength": e["strength"],
                })
                available_for_entries -= notional
                logger.info(f"    ✓ {sym} bought ${notional:,.0f}")
            else:
                msg = result.get("message", str(result))
                entry_results.append({"symbol": sym, "status": "failed", "error": msg})
                logger.warning(f"    ✗ {sym} buy failed: {msg}")
        except Exception as ex:
            entry_results.append({"symbol": sym, "status": "error", "error": str(ex)})
            logger.error(f"    ✗ {sym} error: {ex}")
        time.sleep(0.3)

    # ── Step 8: Final Summary ──
    logger.info("\n" + "=" * 60)
    logger.info("  DAILY EXECUTION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Pipeline signals: {report.total_signals}")
    logger.info(f"  Unique tickers: {len(signal_map)} (BUY: {len(buy_signals)}, SELL: {len(sell_signals)})")
    logger.info(f"  Exits executed: {len([r for r in exit_results if r.get('status') == 'closed'])}/{len(exits)}")
    logger.info(f"  Rebalances executed: {len([r for r in rebal_results if r.get('status') == 'ok'])}/{len(rebalances)}")
    logger.info(f"  New entries: {len([r for r in entry_results if r.get('order_id')])}/{len(actual_entries)}")
    logger.info("=" * 60)

    # Save detailed report
    report_path = os.path.join(os.path.dirname(__file__), "pipeline_report.json")
    with open(report_path, "w") as fp:
        json.dump({
            "run_date": datetime.utcnow().isoformat(),
            "pipeline": {
                "status": report.overall_status,
                "total_signals": report.total_signals,
                "buy_signals": len(buy_signals),
                "sell_signals": len(sell_signals),
            },
            "portfolio": {
                "equity": equity,
                "positions_before": len(positions),
                "exits": exit_results,
                "rebalances": rebal_results,
                "entries": entry_results,
            },
            "signal_map": {k: {"direction": v["direction"], "strength": v["strength"]}
                          for k, v in signal_map.items()},
        }, fp, indent=2, default=str)
    logger.info(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
