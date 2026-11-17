"""
Moving-average crossover trading bot (Alpaca PAPER account), multi-stock version.

Each run:
  1. Gets Alpaca's list of the most active stocks today (by volume), drops
     excluded symbols, cheap stocks and ETFs, and adds every stock already held.
  2. Pulls daily prices for all of them from Yahoo Finance (completed days only).
  3. Looks for a real crossover between the last two completed days:
       short MA crosses ABOVE long MA -> BUY  (POSITION_SIZE_PCT of account, whole shares)
       short MA crosses BELOW long MA -> SELL (the whole position, if held)
  4. Places sells first, then buys (most active first) up to MAX_POSITIONS.
  5. Writes scan_results.json with what it saw for every stock.

Usage:
  python trading_bot.py                      # normal run (uses DRY_RUN from config.py)
  python trading_bot.py --dry-run            # never place an order, just report
  python trading_bot.py --dry-run --symbols AAPL,MSFT,NVDA   # test without the screener
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

import config

NY_TZ = "America/New_York"
BASE_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
def setup_logging() -> logging.Logger:
    logger = logging.getLogger("ma_bot")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s ET  %(levelname)-7s  %(message)s", "%Y-%m-%d %H:%M:%S")
    # Always log in New York time, even on cloud servers that run in UTC.
    fmt.converter = lambda secs: datetime.fromtimestamp(secs, ZoneInfo(NY_TZ)).timetuple()
    handlers = [logging.FileHandler(BASE_DIR / config.LOG_FILE, encoding="utf-8")]
    if sys.stdout is not None:  # no console when run hidden by Task Scheduler (pythonw)
        handlers.append(logging.StreamHandler(sys.stdout))
    for handler in handlers:
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


log = setup_logging()


def validate_config() -> None:
    if config.SHORT_MA_PERIOD >= config.LONG_MA_PERIOD:
        raise ValueError("SHORT_MA_PERIOD must be smaller than LONG_MA_PERIOD")
    if config.LOOKBACK_DAYS < config.LONG_MA_PERIOD + 1:
        raise ValueError("LOOKBACK_DAYS must be at least LONG_MA_PERIOD + 1")
    if not 0 < config.POSITION_SIZE_PCT <= 100:
        raise ValueError("POSITION_SIZE_PCT must be between 0 and 100")
    if not 1 <= config.MOST_ACTIVE_TOP <= 100:
        raise ValueError("MOST_ACTIVE_TOP must be between 1 and 100")


def load_state() -> dict:
    path = BASE_DIR / config.STATE_FILE
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            log.warning("State file was unreadable; starting fresh")
    return {}


def save_state(state: dict) -> None:
    (BASE_DIR / config.STATE_FILE).write_text(json.dumps(state, indent=2))


# ---------------------------------------------------------------------------
# Alpaca
# ---------------------------------------------------------------------------
def get_keys() -> tuple[str, str]:
    load_dotenv(BASE_DIR / ".env")
    return os.getenv("ALPACA_API_KEY", "").strip(), os.getenv("ALPACA_SECRET_KEY", "").strip()


def get_trading_client():
    from alpaca.trading.client import TradingClient

    key, secret = get_keys()
    if not key or not secret:
        raise RuntimeError("ALPACA_API_KEY / ALPACA_SECRET_KEY are missing from .env")
    # paper=True forces the paper-trading endpoint. This bot never touches a live account.
    return TradingClient(key, secret, paper=True)


def market_trades_today(client) -> bool:
    """True if the stock market is open now or opens later today (False on weekends/holidays)."""
    clock = client.get_clock()
    if clock.is_open:
        return True
    today_ny = pd.Timestamp.now(tz=NY_TZ).date()
    return pd.Timestamp(clock.next_open).tz_convert(NY_TZ).date() == today_ny


def most_active_symbols(top: int) -> list[str]:
    from alpaca.data.historical.screener import ScreenerClient
    from alpaca.data.requests import MostActivesRequest

    key, secret = get_keys()
    screener = ScreenerClient(key, secret)
    result = screener.get_most_actives(MostActivesRequest(top=top, by="volume"))
    return [s.symbol for s in result.most_actives]


def held_positions(client) -> tuple[dict[str, float], dict[str, float]]:
    """(shares held, market value) per symbol."""
    positions = client.get_all_positions()
    return ({p.symbol: float(p.qty) for p in positions},
            {p.symbol: float(p.market_value or 0) for p in positions})


def asset_is_tradable(client, symbol: str) -> tuple[bool, str]:
    try:
        a = client.get_asset(symbol)
    except Exception as exc:  # noqa: BLE001
        return False, f"asset lookup failed ({exc})"
    if not a.tradable or str(a.status).lower().endswith("inactive"):
        return False, "not tradable on Alpaca"
    if "OTC" in str(a.exchange).upper():
        return False, "OTC stock"
    return True, ""


def place_order(client, symbol: str, side: str, qty: float):
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest

    order = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.BUY if side == "BUY" else OrderSide.SELL,
        time_in_force=TimeInForce.DAY,
    )
    return client.submit_order(order)


# ---------------------------------------------------------------------------
# Market data + signals
# ---------------------------------------------------------------------------
def fetch_closes(symbols: list[str]) -> pd.DataFrame:
    """Daily closes (one column per symbol) for completed trading days only."""
    df = pd.DataFrame()
    for attempt in range(1, 4):
        try:
            df = yf.download(symbols, period="1y", interval="1d", auto_adjust=True,
                             group_by="ticker", threads=True, progress=False)
        except Exception as exc:  # noqa: BLE001
            log.warning("Yahoo Finance attempt %d failed: %s", attempt, exc)
        if not df.empty:
            break
        if attempt < 3:
            time.sleep(10 * attempt)
    if df.empty:
        raise RuntimeError("No price data returned from Yahoo Finance")

    if isinstance(df.columns, pd.MultiIndex):
        closes = df.xs("Close", axis=1, level=1)
    else:
        closes = df[["Close"]].rename(columns={"Close": symbols[0]})

    # Drop today's still-forming bar so signals only use finished days.
    now_ny = pd.Timestamp.now(tz=NY_TZ)
    last = pd.Timestamp(closes.index[-1])
    last_date = (last.tz_convert(NY_TZ) if last.tzinfo else last).date()
    if last_date == now_ny.date() and now_ny.hour < 16:
        closes = closes.iloc[:-1]
    return closes


def analyse(series: pd.Series) -> dict | None:
    """Moving averages and crossover signal for one stock, or None if not enough data."""
    s = series.dropna().tail(config.LOOKBACK_DAYS)
    if len(s) < config.LONG_MA_PERIOD + 1:
        return None
    short = s.rolling(config.SHORT_MA_PERIOD).mean()
    long_ = s.rolling(config.LONG_MA_PERIOD).mean()
    prev_diff = short.iloc[-2] - long_.iloc[-2]
    curr_diff = short.iloc[-1] - long_.iloc[-1]
    signal = None
    if prev_diff <= 0 < curr_diff:
        signal = "BUY"
    elif prev_diff >= 0 > curr_diff:
        signal = "SELL"
    return {
        "date": pd.Timestamp(s.index[-1]).strftime("%Y-%m-%d"),
        "close": round(float(s.iloc[-1]), 2),
        "sma_short": round(float(short.iloc[-1]), 2),
        "sma_long": round(float(long_.iloc[-1]), 2),
        "gap_pct": round(float(curr_diff / long_.iloc[-1] * 100), 2),
        "signal": signal,
    }


_quote_types: dict[str, str] = {}


def quote_type(symbol: str) -> str:
    if symbol not in _quote_types:
        try:
            _quote_types[symbol] = str(yf.Ticker(symbol).info.get("quoteType", "UNKNOWN")).upper()
        except Exception:  # noqa: BLE001
            _quote_types[symbol] = "UNKNOWN"
    return _quote_types[symbol]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(dry_run: bool, symbols_override: list[str] | None) -> None:
    validate_config()
    log.info("---- Run start: most-active scan  SMA%d/SMA%d  size=%.1f%%  max=%d  cap=%.0f%%  dry_run=%s ----",
             config.SHORT_MA_PERIOD, config.LONG_MA_PERIOD, config.POSITION_SIZE_PCT,
             config.MAX_POSITIONS, config.MAX_INVESTED_PCT, dry_run)

    key, secret = get_keys()
    client = get_trading_client() if (key and secret) else None
    if client is None and not dry_run:
        raise RuntimeError("ALPACA_API_KEY / ALPACA_SECRET_KEY are missing from .env")

    if not dry_run and not market_trades_today(client):
        log.info("Market is not open for trading now (weekend, holiday, or after hours). Nothing to do.")
        return

    # 1. Which stocks to look at
    if symbols_override:
        active = symbols_override
        log.info("Using symbols from the command line: %s", ", ".join(active))
    else:
        active = most_active_symbols(config.MOST_ACTIVE_TOP)
        log.info("Most active today: %d stocks", len(active))
    excluded = {s.upper() for s in config.EXCLUDE_SYMBOLS}
    active = [s for s in active if s.upper() not in excluded and s.isalpha()]

    held, held_value = held_positions(client) if client else ({}, {})
    if held:
        log.info("Holding %d stock(s): %s", len(held), ", ".join(f"{s} x{q:g}" for s, q in held.items()))
    universe = list(dict.fromkeys(active + list(held)))

    # 2. Prices + signals
    closes = fetch_closes(universe)
    rows, buys, sells = [], [], []
    for rank, sym in enumerate(universe, start=1):
        row = {"symbol": sym, "rank": rank if sym in active else None, "held_qty": held.get(sym, 0.0),
               "close": None, "sma_short": None, "sma_long": None, "gap_pct": None,
               "signal": "", "action": "", "note": ""}
        rows.append(row)
        if sym not in closes.columns:
            row["note"] = "no price data"
            continue
        a = analyse(closes[sym])
        if a is None:
            row["note"] = "not enough price history"
            continue
        row.update({k: a[k] for k in ("close", "sma_short", "sma_long", "gap_pct")})
        row["signal"] = a["signal"] or ""
        row["date"] = a["date"]
        if sym not in held and a["close"] < config.MIN_PRICE:
            row["signal"] = ""
            row["note"] = f"under ${config.MIN_PRICE:g}" + (f"; ignored its {a['signal'].lower()} crossover" if a["signal"] else "")
            continue
        if a["signal"] == "SELL":
            sells.append(row)
        elif a["signal"] == "BUY":
            buys.append(row)

    log.info("Scanned %d stocks: %d buy signal(s), %d sell signal(s)", len(universe), len(buys), len(sells))
    for r in buys + sells:
        log.info("%s signal: %s  close=%.2f  SMA%d=%.2f  SMA%d=%.2f", r["signal"], r["symbol"], r["close"],
                 config.SHORT_MA_PERIOD, r["sma_short"], config.LONG_MA_PERIOD, r["sma_long"])

    state = load_state()

    def already_done(r) -> bool:
        return state.get(r["symbol"], {}).get("last_signal_date") == r.get("date")

    def remember(r) -> None:
        state[r["symbol"]] = {"last_signal_date": r.get("date"), "last_signal": r["signal"]}

    # 3. Sells first
    for r in sells:
        sym, qty = r["symbol"], held.get(r["symbol"], 0.0)
        if already_done(r):
            r["action"] = "skipped"; r["note"] = "already acted on this crossover"
        elif qty <= 0 and not config.ALLOW_SHORT:
            r["action"] = "none"; r["note"] = "not held"
        elif dry_run:
            r["action"] = "would sell"; r["note"] = f"{qty:g} shares"
            log.info("[DRY RUN] Would sell %g %s. No order sent.", qty, sym)
        else:
            order = place_order(client, sym, "SELL", qty)
            r["action"] = "sold"; r["note"] = f"{qty:g} shares"
            log.info("Order submitted: SELL %g %s  id=%s  status=%s", qty, sym, order.id, order.status)
            held.pop(sym, None)
            held_value.pop(sym, None)
            remember(r)

    # 4. Buys, most active first, within position and cash limits
    account = client.get_account() if client else None
    equity = float(account.equity) if account else 100000.0
    cash = float(account.cash) if account else equity
    budget = equity * config.POSITION_SIZE_PCT / 100
    open_slots = config.MAX_POSITIONS - len(held)
    invested = sum(held_value.values())
    invest_cap = equity * config.MAX_INVESTED_PCT / 100
    for r in sorted(buys, key=lambda x: x["rank"] or 999):
        sym = r["symbol"]
        if already_done(r):
            r["action"] = "skipped"; r["note"] = "already acted on this crossover"; continue
        if sym in held:
            r["action"] = "skipped"; r["note"] = "already held"; continue
        if open_slots <= 0:
            r["action"] = "skipped"; r["note"] = f"already holding {config.MAX_POSITIONS} stocks"; continue
        if not config.ALLOW_ETFS and quote_type(sym) != "EQUITY":
            r["action"] = "skipped"; r["note"] = f"not a company stock ({quote_type(sym).lower()})"; continue
        if client:
            ok, why = asset_is_tradable(client, sym)
            if not ok:
                r["action"] = "skipped"; r["note"] = why; continue
        qty = math.floor(budget / r["close"])
        if qty < 1:
            r["action"] = "skipped"; r["note"] = f"one share costs more than ${budget:,.0f}"; continue
        cost = qty * r["close"]
        if cost > cash:
            r["action"] = "skipped"; r["note"] = "not enough cash"; continue
        if invested + cost > invest_cap:
            r["action"] = "skipped"
            r["note"] = f"would put more than {config.MAX_INVESTED_PCT:g}% of the account in stocks"
            continue
        if dry_run:
            r["action"] = "would buy"; r["note"] = f"{qty} shares (~${cost:,.0f})"
            log.info("[DRY RUN] Would buy %d %s (~$%.0f). No order sent.", qty, sym, cost)
        else:
            order = place_order(client, sym, "BUY", qty)
            r["action"] = "bought"; r["note"] = f"{qty} shares (~${cost:,.0f})"
            log.info("Order submitted: BUY %d %s  id=%s  status=%s", qty, sym, order.id, order.status)
            remember(r)
        cash -= cost
        invested += cost
        open_slots -= 1

    if not buys and not sells:
        log.info("No crossovers today. Nothing to do.")
    if not dry_run:
        save_state(state)

    # 5. Save what this run saw
    run_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for r in rows:
        r["run_at"] = run_at
        r["dry_run"] = dry_run
        r.pop("date", None)
    (BASE_DIR / config.SCAN_FILE).write_text(json.dumps(rows, indent=1))
    log.info("Saved scan of %d stocks to %s", len(rows), config.SCAN_FILE)


def main() -> None:
    parser = argparse.ArgumentParser(description="SMA crossover bot for Alpaca paper trading")
    parser.add_argument("--dry-run", action="store_true", help="report only, never place orders")
    parser.add_argument("--symbols", help="comma-separated symbols to scan instead of the most-active list")
    args = parser.parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",")] if args.symbols else None
    try:
        run(dry_run=args.dry_run or config.DRY_RUN, symbols_override=symbols)
    except Exception as exc:  # log and exit non-zero so a scheduler can notice
        log.exception("Run failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
