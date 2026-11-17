"""
Moving-average crossover trading bot (Alpaca PAPER account).

Each run:
  1. Pulls the last LOOKBACK_DAYS daily bars for SYMBOL from Yahoo Finance.
  2. Computes the short and long simple moving averages.
  3. Compares the last two completed days. A crossover only counts when the
     short MA was on one side of the long MA yesterday and on the other side today.
  4. Golden cross (short crosses above long)  -> market BUY  TRADE_QTY shares.
     Death cross  (short crosses below long)  -> market SELL TRADE_QTY shares.

Usage:
  python trading_bot.py            # normal run (uses DRY_RUN from config.py)
  python trading_bot.py --dry-run  # never place an order, just report
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

import config

BASE_DIR = Path(__file__).resolve().parent
NY_TZ = "America/New_York"


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
def setup_logging() -> logging.Logger:
    logger = logging.getLogger("ma_bot")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s", "%Y-%m-%d %H:%M:%S")
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
    if config.TRADE_QTY <= 0:
        raise ValueError("TRADE_QTY must be positive")


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
# Market data + signal
# ---------------------------------------------------------------------------
def fetch_prices(symbol: str, days: int) -> pd.DataFrame:
    """Return the last `days` COMPLETED daily bars (today's bar is dropped while the market is open)."""
    # Ask for a year so we always get at least `days` trading days back.
    # Yahoo sometimes throttles cloud servers, so retry a few times before giving up.
    df = pd.DataFrame()
    for attempt in range(1, 4):
        try:
            df = yf.Ticker(symbol).history(period="1y", interval="1d", auto_adjust=True)
        except Exception as exc:
            log.warning("Yahoo Finance attempt %d failed: %s", attempt, exc)
        if not df.empty:
            break
        if attempt < 3:
            time.sleep(10 * attempt)
    if df.empty:
        raise RuntimeError(f"No price data returned from Yahoo Finance for {symbol}")

    # Drop today's still-forming bar so the signal only uses finished days.
    now_ny = pd.Timestamp.now(tz=NY_TZ)
    last_bar_date = df.index[-1].tz_convert(NY_TZ).date()
    if last_bar_date == now_ny.date() and now_ny.hour < 16:
        df = df.iloc[:-1]

    return df[["Close"]].tail(days).copy()


def detect_crossover(df: pd.DataFrame) -> tuple[str | None, pd.DataFrame]:
    """Add MA columns and return ('BUY' | 'SELL' | None, df)."""
    df["sma_short"] = df["Close"].rolling(config.SHORT_MA_PERIOD).mean()
    df["sma_long"] = df["Close"].rolling(config.LONG_MA_PERIOD).mean()
    valid = df.dropna()
    if len(valid) < 2:
        raise RuntimeError(
            f"Only {len(valid)} day(s) with both averages; need 2. Increase LOOKBACK_DAYS."
        )

    prev, curr = valid.iloc[-2], valid.iloc[-1]
    prev_diff = prev["sma_short"] - prev["sma_long"]
    curr_diff = curr["sma_short"] - curr["sma_long"]

    if prev_diff <= 0 < curr_diff:
        return "BUY", valid
    if prev_diff >= 0 > curr_diff:
        return "SELL", valid
    return None, valid


# ---------------------------------------------------------------------------
# Alpaca
# ---------------------------------------------------------------------------
def get_trading_client():
    from alpaca.trading.client import TradingClient

    load_dotenv(BASE_DIR / ".env")
    key = os.getenv("ALPACA_API_KEY", "").strip()
    secret = os.getenv("ALPACA_SECRET_KEY", "").strip()
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


def current_position_qty(client, symbol: str) -> float:
    from alpaca.common.exceptions import APIError

    try:
        return float(client.get_open_position(symbol).qty)
    except APIError:
        return 0.0  # no open position


def place_order(client, symbol: str, side: str, qty: int):
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
# Main
# ---------------------------------------------------------------------------
def run(dry_run: bool) -> None:
    validate_config()
    symbol = config.SYMBOL.upper()
    log.info("---- Run start: %s  SMA%d/SMA%d  qty=%d  dry_run=%s ----",
             symbol, config.SHORT_MA_PERIOD, config.LONG_MA_PERIOD, config.TRADE_QTY, dry_run)

    client = None
    if not dry_run:
        client = get_trading_client()
        if not market_trades_today(client):
            log.info("Market is closed today (weekend or holiday). Nothing to do.")
            return

    df = fetch_prices(symbol, config.LOOKBACK_DAYS)
    signal, valid = detect_crossover(df)

    last = valid.iloc[-1]
    signal_date = valid.index[-1].strftime("%Y-%m-%d")
    log.info("Latest completed day %s: close=%.2f  SMA%d=%.2f  SMA%d=%.2f",
             signal_date, last["Close"], config.SHORT_MA_PERIOD, last["sma_short"],
             config.LONG_MA_PERIOD, last["sma_long"])

    if signal is None:
        trend = "above" if last["sma_short"] > last["sma_long"] else "below"
        log.info("No crossover. SMA%d is %s SMA%d. Nothing to do.",
                 config.SHORT_MA_PERIOD, trend, config.LONG_MA_PERIOD)
        return

    log.info("%s signal: SMA%d crossed %s SMA%d on %s",
             signal, config.SHORT_MA_PERIOD, "above" if signal == "BUY" else "below",
             config.LONG_MA_PERIOD, signal_date)

    # Don't act on the same crossover twice if the bot runs more than once a day.
    state = load_state()
    if state.get(symbol, {}).get("last_signal_date") == signal_date:
        log.info("Already acted on the %s crossover of %s. Skipping.", signal, signal_date)
        return

    if dry_run:
        log.info("[DRY RUN] Would place market %s for %d %s. No order sent.",
                 signal, config.TRADE_QTY, symbol)
        return

    held = current_position_qty(client, symbol)
    log.info("Current %s position: %s shares", symbol, held)

    if signal == "BUY" and held > 0 and not config.ALLOW_ADD_TO_POSITION:
        log.info("Already holding %s; skipping buy (ALLOW_ADD_TO_POSITION=False).", symbol)
    elif signal == "SELL" and held < config.TRADE_QTY and not config.ALLOW_SHORT:
        log.info("Not enough shares to sell (have %s); skipping (ALLOW_SHORT=False).", held)
    else:
        order = place_order(client, symbol, signal, config.TRADE_QTY)
        log.info("Order submitted: %s %d %s  id=%s  status=%s",
                 signal, config.TRADE_QTY, symbol, order.id, order.status)

    state[symbol] = {"last_signal_date": signal_date, "last_signal": signal}
    save_state(state)


def main() -> None:
    parser = argparse.ArgumentParser(description="SMA crossover bot for Alpaca paper trading")
    parser.add_argument("--dry-run", action="store_true", help="report only, never place orders")
    args = parser.parse_args()
    try:
        run(dry_run=args.dry_run or config.DRY_RUN)
    except Exception as exc:  # log and exit non-zero so a scheduler can notice
        log.exception("Run failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
