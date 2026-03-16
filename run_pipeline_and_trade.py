"""
GB Trading — Daily Portfolio Manager + Signal Executor (v2)
===========================================================
Runs ALL 22 strategies, generates signals for ALL asset classes,
and executes on Alpaca Paper Trading with FULL capabilities:

- Stocks (long + short)
- ETFs (long + short where shortable)
- Crypto (long only — Alpaca doesn't support crypto shorting)
- FX proxies (UUP, FXE, FXY, etc.)
- Commodity proxies (GLD, SLV, USO, etc.)
- Fixed income proxies (TLT, IEF, SHY, etc.)
- Volatility instruments (VXX, SVXY, UVXY)

Short selling is enabled for all shortable assets.
"""

import os
import sys
import json
import time
import logging
import requests
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))

from strategies.automation.daily_pipeline import DailyPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("gb_trading.executor")

# ── Configuration ──────────────────────────────────────────────────────────
TARGET_WEIGHT = 0.04           # 4% per position (target for new entries)
MAX_WEIGHT = 0.08              # 8% hard cap before trimming
MIN_WEIGHT = 0.02              # 2% floor before topping up
REBALANCE_THRESHOLD = 0.015    # 1.5% drift triggers rebalance
MAX_POSITIONS = 30             # Maximum concurrent positions (increased for multi-asset)
MIN_ORDER_VALUE = 200          # Minimum order notional ($)
CASH_RESERVE_PCT = 0.10        # Keep 10% cash reserve
STOP_LOSS_PCT = -0.10          # -10% stop loss

# Crypto tickers need special handling (different symbol format on Alpaca)
CRYPTO_TICKERS = {"BTC-USD", "ETH-USD", "SOL-USD"}
CRYPTO_ALPACA_MAP = {
    "BTC-USD": "BTC/USD",
    "ETH-USD": "BTC/USD",  # Will be corrected below
    "SOL-USD": "SOL/USD",
}
# Fix the mapping
CRYPTO_ALPACA_MAP["ETH-USD"] = "ETH/USD"

# Tickers that CANNOT be shorted on Alpaca
NON_SHORTABLE = {
    "USO", "UNG", "DBA", "DBB",      # Commodity ETFs
    "UUP", "FXE", "FXY", "FXB", "FXA",  # FX ETFs
    "VXX", "SVXY", "UVXY",           # Volatility ETNs
    "BTC/USD", "ETH/USD", "SOL/USD", # Crypto
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
    resp = requests.get(f"{ALPACA_BASE}{endpoint}", headers=HEADERS, timeout=15)
    resp.raise_for_status()
    return resp.json()


def alpaca_post(endpoint, payload):
    resp = requests.post(f"{ALPACA_BASE}{endpoint}", headers=HEADERS, json=payload, timeout=15)
    return resp.json(), resp.status_code


def alpaca_delete(endpoint):
    resp = requests.delete(f"{ALPACA_BASE}{endpoint}", headers=HEADERS, timeout=15)
    return resp.status_code


def get_account():
    return alpaca_get("/v2/account")


def get_positions():
    return alpaca_get("/v2/positions")


def cancel_all_orders():
    return alpaca_delete("/v2/orders")


def submit_market_order(symbol, notional, side):
    """Submit a market order using notional amount (fractional shares). Long only."""
    payload = {
        "symbol": symbol,
        "notional": str(round(notional, 2)),
        "side": side,
        "type": "market",
        "time_in_force": "day",
    }
    return alpaca_post("/v2/orders", payload)


def get_last_price(symbol):
    """Get the last trade price for a symbol."""
    try:
        resp = requests.get(
            f"https://data.alpaca.markets/v2/stocks/{symbol}/trades/latest",
            headers=HEADERS, timeout=10,
        )
        if resp.status_code == 200:
            return float(resp.json().get("trade", {}).get("p", 0))
    except Exception:
        pass
    return 0


def submit_short_order(symbol, notional):
    """Submit a short sell order using whole shares (Alpaca requires integer qty for shorts)."""
    price = get_last_price(symbol)
    if price <= 0:
        return {"message": f"Could not get price for {symbol}"}, 400
    qty = int(notional / price)
    if qty <= 0:
        return {"message": f"Qty too small for {symbol} at ${price:.2f}"}, 400
    payload = {
        "symbol": symbol,
        "qty": str(qty),
        "side": "sell",
        "type": "market",
        "time_in_force": "day",
    }
    return alpaca_post("/v2/orders", payload)


def submit_qty_order(symbol, qty, side):
    """Submit a market order using quantity (for shorts and crypto)."""
    payload = {
        "symbol": symbol,
        "qty": str(abs(qty)),
        "side": side,
        "type": "market",
        "time_in_force": "day",
    }
    return alpaca_post("/v2/orders", payload)


def close_position(symbol):
    return alpaca_delete(f"/v2/positions/{symbol}")


def check_shortable(symbol):
    """Check if a symbol is shortable on Alpaca."""
    if symbol in NON_SHORTABLE:
        return False
    try:
        resp = requests.get(
            f"{ALPACA_BASE}/v2/assets/{symbol}",
            headers=HEADERS, timeout=10,
        )
        if resp.status_code == 200:
            asset = resp.json()
            return asset.get("shortable", False) and asset.get("easy_to_borrow", False)
    except Exception:
        pass
    return False


def to_alpaca_symbol(ticker):
    """Convert pipeline ticker to Alpaca symbol format."""
    if ticker in CRYPTO_ALPACA_MAP:
        return CRYPTO_ALPACA_MAP[ticker]
    return ticker


def from_alpaca_symbol(symbol):
    """Convert Alpaca symbol back to pipeline ticker format."""
    reverse_map = {v: k for k, v in CRYPTO_ALPACA_MAP.items()}
    return reverse_map.get(symbol, symbol)


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
    NO EXCLUSIONS — all tickers from all strategies are included.
    Returns: {ticker: {"direction": "long"|"short", "strength": float, "strategies": [str], "asset_class": str}}
    """
    ticker_signals = defaultdict(lambda: {
        "long_strength": 0, "short_strength": 0,
        "strategies": [], "asset_class": "equity",
    })

    # Asset class classification
    etf_tickers = {
        "SPY", "QQQ", "IWM", "DIA",
        "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC",
    }
    fi_tickers = {"TLT", "IEF", "SHY", "LQD", "HYG", "TIP", "AGG"}
    commodity_tickers = {"GLD", "SLV", "USO", "UNG", "DBA", "DBB"}
    fx_tickers = {"UUP", "FXE", "FXY", "FXB", "FXA"}
    vol_tickers = {"VXX", "SVXY", "UVXY"}
    crypto_tickers = {"BTC-USD", "ETH-USD", "SOL-USD"}

    def classify(t):
        if t in etf_tickers: return "etf"
        if t in fi_tickers: return "fixed_income"
        if t in commodity_tickers: return "commodity"
        if t in fx_tickers: return "fx"
        if t in vol_tickers: return "volatility"
        if t in crypto_tickers: return "crypto"
        return "equity"

    for sig in raw_signals:
        ticker = sig.get("ticker", "")
        if not ticker:
            continue
        direction = sig.get("direction", "")
        strength = abs(sig.get("signal_strength", 0))
        strategy = sig.get("strategy", "unknown")

        ticker_signals[ticker]["asset_class"] = classify(ticker)

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
        if abs(net) > 0.01:
            signal_map[ticker] = {
                "direction": "long" if net > 0 else "short",
                "strength": abs(net),
                "strategies": data["strategies"],
                "asset_class": data["asset_class"],
            }

    return signal_map


# ── Portfolio Management ───────────────────────────────────────────────────

def analyze_portfolio(positions, equity, signal_map, cash):
    """
    Full portfolio management with SHORT SELLING support.
    
    Strategy:
    1. EXIT long positions with SELL signals or stop-loss
    2. COVER short positions with BUY signals or stop-loss
    3. REBALANCE overweight positions
    4. ENTER new LONG positions with BUY signals
    5. ENTER new SHORT positions with SELL signals (if shortable)
    6. REINFORCE existing positions with excess cash
    """
    exits = []
    entries = []
    rebalances = []

    held = {}
    for p in positions:
        sym = from_alpaca_symbol(p["symbol"])
        mv = abs(float(p["market_value"]))
        qty = float(p["qty"])
        side = "long" if qty > 0 else "short"
        weight = mv / equity if equity > 0 else 0
        pnl_pct = float(p["unrealized_plpc"])
        held[sym] = {
            "alpaca_symbol": p["symbol"],
            "market_value": mv,
            "weight": weight,
            "pnl_pct": pnl_pct,
            "qty": qty,
            "side": side,
        }

    # 1. EXITS: Close positions based on signal reversal or stop-loss
    for sym, pos_data in held.items():
        sig = signal_map.get(sym)
        should_exit = False
        reason = ""

        if pos_data["side"] == "long":
            # Exit long if sell signal
            if sig and sig["direction"] == "short":
                should_exit = True
                reason = f"Signal reversal to SHORT (strength={sig['strength']:.2f})"
            elif pos_data["pnl_pct"] < STOP_LOSS_PCT:
                should_exit = True
                reason = f"Stop loss triggered (P&L={pos_data['pnl_pct']*100:.1f}%)"
        elif pos_data["side"] == "short":
            # Cover short if buy signal
            if sig and sig["direction"] == "long":
                should_exit = True
                reason = f"Signal reversal to LONG (strength={sig['strength']:.2f})"
            elif pos_data["pnl_pct"] < STOP_LOSS_PCT:
                should_exit = True
                reason = f"Stop loss triggered (P&L={pos_data['pnl_pct']*100:.1f}%)"

        if should_exit:
            exits.append({
                "symbol": sym,
                "alpaca_symbol": pos_data["alpaca_symbol"],
                "reason": reason,
                "market_value": pos_data["market_value"],
                "side": pos_data["side"],
            })

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
                    "alpaca_symbol": pos_data["alpaca_symbol"],
                    "action": "trim",
                    "amount": trim_amount,
                    "side": pos_data["side"],
                    "reason": f"Overweight ({pos_data['weight']*100:.1f}% → {TARGET_WEIGHT*100:.1f}%)",
                })

    # 3. NEW ENTRIES: Long and Short
    still_held = set(held.keys()) - exit_symbols
    long_candidates = []
    short_candidates = []

    for ticker, sig in signal_map.items():
        if ticker in still_held:
            continue
        alpaca_sym = to_alpaca_symbol(ticker)
        if sig["direction"] == "long":
            long_candidates.append((ticker, alpaca_sym, sig["strength"], sig["strategies"], sig["asset_class"]))
        elif sig["direction"] == "short":
            short_candidates.append((ticker, alpaca_sym, sig["strength"], sig["strategies"], sig["asset_class"]))

    long_candidates.sort(key=lambda x: x[2], reverse=True)
    short_candidates.sort(key=lambda x: x[2], reverse=True)

    current_count = len(still_held)
    available_slots = MAX_POSITIONS - current_count

    # Allocate slots: 70% long, 30% short
    long_slots = max(1, int(available_slots * 0.7))
    short_slots = available_slots - long_slots

    for ticker, alpaca_sym, strength, strategies, asset_class in long_candidates[:long_slots]:
        if strength > 0.1:
            entries.append({
                "symbol": ticker,
                "alpaca_symbol": alpaca_sym,
                "side": "long",
                "strength": strength,
                "strategies": strategies,
                "asset_class": asset_class,
            })

    for ticker, alpaca_sym, strength, strategies, asset_class in short_candidates[:short_slots]:
        if strength > 0.3:  # Higher threshold for shorts
            entries.append({
                "symbol": ticker,
                "alpaca_symbol": alpaca_sym,
                "side": "short",
                "strength": strength,
                "strategies": strategies,
                "asset_class": asset_class,
            })

    # 4. REINFORCE: Deploy excess cash
    invested_after = sum(d["market_value"] for s, d in held.items() if s not in exit_symbols)
    entry_investment = len(entries) * TARGET_WEIGHT * equity
    invested_after += entry_investment

    target_invested = (1.0 - CASH_RESERVE_PCT) * equity
    cash_to_deploy = target_invested - invested_after

    if cash_to_deploy > MIN_ORDER_VALUE:
        reinforce_candidates = []
        for sym in still_held:
            sig = signal_map.get(sym)
            pos = held[sym]
            # Reinforce long positions with buy signals
            if pos["side"] == "long" and sig and sig["direction"] == "long":
                reinforce_candidates.append((sym, pos["alpaca_symbol"], sig["strength"], pos["market_value"], "long"))
            # Reinforce short positions with sell signals
            elif pos["side"] == "short" and sig and sig["direction"] == "short":
                reinforce_candidates.append((sym, pos["alpaca_symbol"], sig["strength"], pos["market_value"], "short"))

        reinforce_candidates.sort(key=lambda x: x[2], reverse=True)

        if reinforce_candidates:
            equal_share = cash_to_deploy / len(reinforce_candidates)
            for sym, alpaca_sym, strength, current_mv, side in reinforce_candidates:
                max_additional = (MAX_WEIGHT * equity) - current_mv
                share = min(equal_share, max(0, max_additional))
                if share > MIN_ORDER_VALUE:
                    rebalances.append({
                        "symbol": sym,
                        "alpaca_symbol": alpaca_sym,
                        "action": "topup",
                        "amount": round(share, 2),
                        "side": side,
                        "reason": f"Deploy cash (signal={strength:.2f}, cap {MAX_WEIGHT*100:.0f}%)",
                    })

    return exits, entries, rebalances


# ── Main Execution ─────────────────────────────────────────────────────────

def main():
    logger.info("=" * 70)
    logger.info("  GB Trading — Daily Portfolio Manager v2 (ALL ASSETS + SHORTS)")
    logger.info(f"  Date: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    logger.info("=" * 70)

    # ── Step 1: Run the pipeline (ALL 22 strategies) ──
    logger.info("\n[STEP 1] Running daily signal pipeline (ALL 22 strategies)...")
    pipeline = DailyPipeline()
    report = pipeline.run_full_pipeline()
    report_dict = pipeline.get_report_dict()

    logger.info(f"  Pipeline status: {report.overall_status}")
    logger.info(f"  Total signals: {report.total_signals}")

    # Extract and aggregate signals — NO EXCLUSIONS
    raw_signals = extract_signals(report)
    signal_map = build_signal_map(raw_signals)
    logger.info(f"  Unique tickers with signals: {len(signal_map)}")

    # Categorize signals
    buy_signals = {k: v for k, v in signal_map.items() if v["direction"] == "long"}
    sell_signals = {k: v for k, v in signal_map.items() if v["direction"] == "short"}
    logger.info(f"  LONG signals: {len(buy_signals)} tickers")
    logger.info(f"  SHORT signals: {len(sell_signals)} tickers")

    # Log by asset class
    by_class = defaultdict(list)
    for ticker, sig in signal_map.items():
        by_class[sig["asset_class"]].append(f"{ticker}({sig['direction'][0].upper()})")
    for ac, tickers in sorted(by_class.items()):
        logger.info(f"    {ac:15s}: {len(tickers)} signals — {', '.join(tickers[:10])}")

    if report.overall_status == "FAILED" and report.total_signals == 0:
        logger.error("Pipeline FAILED with 0 signals. Aborting.")
        for stage in report.stages:
            if stage.errors:
                logger.error(f"  {stage.stage.value}: {stage.errors}")
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

    long_positions = [p for p in positions if float(p["qty"]) > 0]
    short_positions = [p for p in positions if float(p["qty"]) < 0]
    logger.info(f"  Positions: {len(positions)} total ({len(long_positions)} long, {len(short_positions)} short)")

    for p in sorted(positions, key=lambda x: abs(float(x["market_value"])), reverse=True):
        w = abs(float(p["market_value"])) / equity * 100 if equity > 0 else 0
        pnl = float(p["unrealized_plpc"]) * 100
        side = "LONG" if float(p["qty"]) > 0 else "SHORT"
        logger.info(f"    {p['symbol']:8s} {side:5s}: ${abs(float(p['market_value'])):>10,.0f} ({w:.1f}%)  P&L: {pnl:>+.2f}%")

    # ── Step 3: Cancel any open orders ──
    logger.info("\n[STEP 3] Cancelling open orders...")
    cancel_all_orders()
    time.sleep(1)

    # ── Step 4: Analyze portfolio and determine actions ──
    logger.info("\n[STEP 4] Analyzing portfolio vs signals...")
    exits, entries, rebalances = analyze_portfolio(positions, equity, signal_map, cash)

    logger.info(f"  Exits planned: {len(exits)}")
    for e in exits:
        logger.info(f"    EXIT {e['symbol']} ({e['side']}): {e['reason']}")

    logger.info(f"  Entries planned: {len(entries)}")
    for e in entries:
        logger.info(f"    ENTER {e['symbol']} ({e['side']}, {e['asset_class']}): strength={e['strength']:.3f}")

    logger.info(f"  Rebalances planned: {len(rebalances)}")
    for r in rebalances:
        logger.info(f"    {r['action'].upper()} {r['symbol']} ({r['side']}): ${r['amount']:,.0f} ({r['reason']})")

    # ── Step 5: Execute exits ──
    logger.info("\n[STEP 5] Executing exits...")
    exit_results = []
    for e in exits:
        sym = e["alpaca_symbol"]
        logger.info(f"  Closing {sym} ({e['side']})...")
        try:
            status = close_position(sym)
            if status in (200, 204):
                exit_results.append({"symbol": sym, "side": e["side"], "status": "closed"})
                logger.info(f"    ✓ {sym} closed")
            else:
                exit_results.append({"symbol": sym, "status": "failed", "error": f"HTTP {status}"})
                logger.warning(f"    ✗ {sym} close failed: HTTP {status}")
        except Exception as ex:
            exit_results.append({"symbol": sym, "status": "error", "error": str(ex)})
            logger.error(f"    ✗ {sym} error: {ex}")
        time.sleep(0.3)

    if exits:
        time.sleep(2)

    # ── Step 6: Execute rebalances ──
    logger.info("\n[STEP 6] Executing rebalances...")
    rebal_results = []
    for r in rebalances:
        sym = r["alpaca_symbol"]
        action = r["action"]
        amount = r["amount"]

        if action == "trim":
            side = "sell" if r["side"] == "long" else "buy"
        else:  # topup
            side = "buy" if r["side"] == "long" else "sell"

        logger.info(f"  {action.upper()} {sym} ({r['side']}): {side} ${amount:,.0f}...")

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

    # Refresh account
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
        sym = e["alpaca_symbol"]
        notional = min(entry_size, available_for_entries)
        if notional < MIN_ORDER_VALUE:
            logger.info(f"  SKIP {sym}: insufficient buying power (${notional:,.0f})")
            continue

        if e["side"] == "long":
            logger.info(f"  BUY {sym} (LONG, {e['asset_class']}): ${notional:,.0f} (strength={e['strength']:.3f})...")
            try:
                result, status = submit_market_order(sym, notional, "buy")
                if status in (200, 201):
                    entry_results.append({
                        "symbol": sym, "side": "long", "notional": notional,
                        "order_id": result.get("id"), "asset_class": e["asset_class"],
                    })
                    available_for_entries -= notional
                    logger.info(f"    ✓ {sym} LONG ${notional:,.0f}")
                else:
                    msg = result.get("message", str(result))
                    entry_results.append({"symbol": sym, "side": "long", "status": "failed", "error": msg})
                    logger.warning(f"    ✗ {sym} buy failed: {msg}")
            except Exception as ex:
                entry_results.append({"symbol": sym, "side": "long", "status": "error", "error": str(ex)})
                logger.error(f"    ✗ {sym} error: {ex}")

        elif e["side"] == "short":
            # Check if shortable
            is_shortable = check_shortable(sym)
            if not is_shortable:
                logger.info(f"  SKIP {sym}: not shortable on Alpaca")
                entry_results.append({"symbol": sym, "side": "short", "status": "skipped", "error": "not shortable"})
                continue

            logger.info(f"  SELL {sym} (SHORT, {e['asset_class']}): ${notional:,.0f} (strength={e['strength']:.3f})...")
            try:
                result, status = submit_short_order(sym, notional)
                if status in (200, 201):
                    entry_results.append({
                        "symbol": sym, "side": "short", "notional": notional,
                        "order_id": result.get("id"), "asset_class": e["asset_class"],
                    })
                    available_for_entries -= notional
                    logger.info(f"    ✓ {sym} SHORT ${notional:,.0f}")
                else:
                    msg = result.get("message", str(result))
                    entry_results.append({"symbol": sym, "side": "short", "status": "failed", "error": msg})
                    logger.warning(f"    ✗ {sym} short failed: {msg}")
            except Exception as ex:
                entry_results.append({"symbol": sym, "side": "short", "status": "error", "error": str(ex)})
                logger.error(f"    ✗ {sym} error: {ex}")

        time.sleep(0.3)

    # ── Step 8: Final Summary ──
    logger.info("\n" + "=" * 70)
    logger.info("  DAILY EXECUTION SUMMARY (v2 — ALL ASSETS + SHORTS)")
    logger.info("=" * 70)
    logger.info(f"  Pipeline signals: {report.total_signals}")
    logger.info(f"  Unique tickers: {len(signal_map)} (LONG: {len(buy_signals)}, SHORT: {len(sell_signals)})")

    # By asset class
    for ac, tickers in sorted(by_class.items()):
        logger.info(f"    {ac:15s}: {len(tickers)} signals")

    successful_exits = len([r for r in exit_results if r.get("status") == "closed"])
    successful_rebals = len([r for r in rebal_results if r.get("status") == "ok"])
    successful_entries = len([r for r in entry_results if r.get("order_id")])
    long_entries = len([r for r in entry_results if r.get("order_id") and r.get("side") == "long"])
    short_entries = len([r for r in entry_results if r.get("order_id") and r.get("side") == "short"])

    logger.info(f"  Exits executed: {successful_exits}/{len(exits)}")
    logger.info(f"  Rebalances executed: {successful_rebals}/{len(rebalances)}")
    logger.info(f"  New entries: {successful_entries}/{len(actual_entries)} (LONG: {long_entries}, SHORT: {short_entries})")
    logger.info("=" * 70)

    # Save detailed report
    report_path = os.path.join(os.path.dirname(__file__), "pipeline_report.json")
    with open(report_path, "w") as fp:
        json.dump({
            "run_date": datetime.utcnow().isoformat(),
            "version": "v2_all_assets_shorts",
            "pipeline": {
                "status": report.overall_status,
                "total_signals": report.total_signals,
                "long_signals": len(buy_signals),
                "short_signals": len(sell_signals),
                "by_asset_class": {ac: len(t) for ac, t in by_class.items()},
            },
            "portfolio": {
                "equity": equity,
                "positions_before": len(positions),
                "exits": exit_results,
                "rebalances": rebal_results,
                "entries": entry_results,
            },
            "signal_map": {
                k: {
                    "direction": v["direction"],
                    "strength": v["strength"],
                    "asset_class": v["asset_class"],
                }
                for k, v in signal_map.items()
            },
        }, fp, indent=2, default=str)
    logger.info(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
