"""
Small-account day trader: the Stocks in Play strategy for an account of about $1,000
(a separate Alpaca account from the $100,000 day trader in sip_bot.py).

Same method as sip_bot.py with the current rules (opening volume at least 2x normal, the first
5-minute candle's breakout, stop 0.1 x ATR moved to the entry price at +1R, 1% risk, at most 20%
of the account per stock and 50% in all, entries until 3:30 PM, everything closed at 3:55 PM),
changed for a small cash account:
  * long only: a cash account cannot sell short (and fractional shares cannot be shorted). Stocks
    are still ranked with the down-moves included, then only the up-moves are traded, as in the
    $1,000 backtest (backtest_daytrade.py "small_frac": +22% vs +8% with whole shares);
  * fractional shares, so a $200 position can hold part of a $300 stock (Alpaca allows fractional
    market and stop orders); a stock that is not fractionable is bought in whole shares or skipped.

Its settings are the SMALL dictionary below, its keys are SMALL_ALPACA_API_KEY /
SMALL_ALPACA_SECRET_KEY, its orders carry client_order_id "sml-YYYYMMDD-...", its log is
small_bot.log and its day plan small_plan.json. It refuses to trade an account worth more than
SMALL["MAX_ACCOUNT"], in case it is ever given the wrong keys.

Usage:
  python small_bot.py --phase day     # pick, enter, protect
  python small_bot.py --phase close   # manage stops, flatten at 3:55 PM
  python small_bot.py --phase plan    # print today's picks (or --day YYYY-MM-DD) and exit
  add --dry-run to log without placing orders.
"""

import argparse
import io
import json
import logging
import math
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv

import config


class C:
    """This bot's own settings (separate from config.py's SIP_ settings, which the $100,000 day trader uses)."""
    ENABLED = True              # False = the scheduled small bot does nothing
    TOP = 20                    # stocks in play ranked each day (then only the up-moves are traded)
    MIN_RVOL = 2.0              # opening volume at least this many times normal
    MIN_AVG_VOLUME = 1_000_000  # 14-day average daily volume (shares)
    MIN_ATR = 0.50              # 14-day average daily range ($)
    STOP_ATR = 0.10             # stop distance as a fraction of the 14-day ATR
    RISK_PCT = 1.0              # max loss at the stop, % of the account, per trade
    MAX_POS_PCT = 20.0          # largest position, % of the account
    MAX_INVESTED_PCT = 50.0     # all open positions, % of the account
    BREAKEVEN_R = 1.0           # move the stop to the entry price at this many R in profit (None = off)
    ENTRY_UNTIL = "15:30"
    DAY_JOB_UNTIL = "15:00"
    FLATTEN_AT = "15:55"
    DRY_RUN = False
    MAX_ACCOUNT = 25_000.0      # refuse to trade a bigger account (wrong keys?)
    LOG_FILE = "small_bot.log"
    PLAN_FILE = "small_plan.json"
    HISTORY_FILE = "small_history.json"   # one row per trading day, for the dashboard


NY = ZoneInfo("America/New_York")
BASE = Path(__file__).resolve().parent
POLL_SECONDS = 20        # while only managing stops
FAST_POLL_SECONDS = 3    # while watching for breakouts


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging() -> logging.Logger:
    lg = logging.getLogger("small_bot")
    lg.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s ET  %(levelname)-7s  %(message)s", "%Y-%m-%d %H:%M:%S")
    fmt.converter = lambda secs: datetime.fromtimestamp(secs, NY).timetuple()
    hs = [logging.FileHandler(BASE / C.LOG_FILE, encoding="utf-8")]
    if sys.stdout is not None:
        hs.append(logging.StreamHandler(sys.stdout))
    for h in hs:
        h.setFormatter(fmt)
        lg.addHandler(h)
    return lg


log = setup_logging()


def now_ny() -> datetime:
    return datetime.now(NY)


def hm(t: datetime) -> str:
    return t.strftime("%H:%M")


def prefix(day: str) -> str:
    return f"sml-{day.replace('-', '')}-"


def entry_cid(day: str, pick: dict) -> str:
    """Entry order id; it carries the aimed-for price in cents so slippage can be measured later."""
    return f"{prefix(day)}e-{pick['symbol']}-a{int(round(pick['level'] * 100))}"


# ---------------------------------------------------------------------------
# Broker wrapper (thin, so tests can swap in a fake)
# ---------------------------------------------------------------------------
class Broker:
    def __init__(self, key: str, secret: str, dry_run: bool):
        from alpaca.trading.client import TradingClient
        self.c = TradingClient(key, secret, paper=True)  # paper only, always
        self.dry = dry_run
        self.fake: list[dict] = []  # orders "placed" in a dry run, so they are not placed again

    def clock(self):
        return self.c.get_clock()

    def account(self) -> dict:
        a = self.c.get_account()
        return {"equity": float(a.equity), "cash": float(a.cash)}

    def positions(self) -> dict:
        return {p.symbol: {"qty": float(p.qty), "market_value": float(p.market_value or 0),
                           "avg_entry_price": float(p.avg_entry_price)} for p in self.c.get_all_positions()}

    def orders_today(self, since: datetime) -> list[dict]:
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest
        rows = self.c.get_orders(GetOrdersRequest(status=QueryOrderStatus.ALL, after=since, limit=500, direction="asc"))
        out = []
        for o in rows:
            out.append({"id": str(o.id), "cid": o.client_order_id or "", "symbol": o.symbol,
                        "side": str(o.side.value if hasattr(o.side, "value") else o.side).lower(),
                        "type": str(o.order_type.value if hasattr(o.order_type, "value") else o.order_type).lower(),
                        "status": str(o.status.value if hasattr(o.status, "value") else o.status).lower(),
                        "qty": float(o.qty or 0), "filled_qty": float(o.filled_qty or 0),
                        "filled_avg_price": float(o.filled_avg_price) if o.filled_avg_price else None,
                        "stop_price": float(o.stop_price) if o.stop_price else None})
        return out + self.fake

    def asset(self, symbol: str) -> dict:
        a = self.c.get_asset(symbol)
        return {"tradable": bool(a.tradable), "shortable": bool(a.shortable),
                "easy_to_borrow": bool(a.easy_to_borrow), "fractionable": bool(getattr(a, "fractionable", False))}

    def stop_order(self, symbol, qty, side, stop_price, cid):
        if self.dry:
            log.info("[DRY RUN] would place %s stop %s x%d @ %.2f (%s)", side, symbol, qty, stop_price, cid)
            self.fake.append({"id": cid, "cid": cid, "symbol": symbol, "side": side, "type": "stop", "status": "new",
                              "qty": qty, "filled_qty": 0.0, "filled_avg_price": None, "stop_price": stop_price})
            return None
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import StopOrderRequest
        return self.c.submit_order(StopOrderRequest(
            symbol=symbol, qty=qty, side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY, stop_price=round(stop_price, 2), client_order_id=cid))

    def move_stop(self, order_id, stop_price, cid):
        """Change a resting stop order's price (Alpaca replaces it with a new order named cid)."""
        if self.dry:
            log.info("[DRY RUN] would move stop %s to %.2f (%s)", order_id, stop_price, cid)
            for o in self.fake:
                if o["id"] == order_id:
                    o["status"] = "replaced"
                    self.fake.append(dict(o, id=cid, cid=cid, status="new", stop_price=stop_price))
                    break
            return None
        from alpaca.trading.requests import ReplaceOrderRequest
        return self.c.replace_order_by_id(order_id, ReplaceOrderRequest(stop_price=round(stop_price, 2),
                                                                        client_order_id=cid))

    def market_order(self, symbol, qty, side, cid):
        if self.dry:
            log.info("[DRY RUN] would place %s market %s x%d (%s)", side, symbol, qty, cid)
            return None
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest
        return self.c.submit_order(MarketOrderRequest(
            symbol=symbol, qty=qty, side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY, client_order_id=cid))

    def cancel(self, order_id):
        if self.dry:
            log.info("[DRY RUN] would cancel order %s", order_id)
            return
        self.c.cancel_order_by_id(order_id)


# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------
def sp500_symbols() -> list[str]:
    html = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                        headers={"User-Agent": "Mozilla/5.0 (trading bot)"}, timeout=30).text
    return sorted({str(s).replace(".", "-") for s in pd.read_html(io.StringIO(html))[0]["Symbol"]})


def daily_stats(symbols: list[str], today: str) -> pd.DataFrame:
    """Per symbol: previous close, 14-day ATR and average volume, all known before today's open."""
    frames = []
    for i in range(0, len(symbols), 100):
        for attempt in range(3):
            df = yf.download(symbols[i:i + 100], period="3mo", interval="1d", auto_adjust=True,
                             group_by="ticker", threads=True, progress=False)
            if not df.empty:
                frames.append(df)
                break
            time.sleep(5 * (attempt + 1))
    raw = pd.concat(frames, axis=1)
    get = lambda f: raw.xs(f, axis=1, level=1).sort_index()  # noqa: E731
    h, l, c, v = get("High"), get("Low"), get("Close"), get("Volume")
    keep = [d.strftime("%Y-%m-%d") < today for d in c.index]  # completed days only
    h, l, c, v = h[keep], l[keep], c[keep], v[keep]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()]).groupby(level=0).max()
    return pd.DataFrame({"prev_close": c.iloc[-1], "atr": tr.rolling(14).mean().iloc[-1],
                         "avg_volume": v.rolling(14).mean().iloc[-1]}).dropna()


def to_alpaca(s: str) -> str:
    return s.replace("-", ".")


def opening_bars(symbols: list[str], start: datetime, end: datetime) -> pd.DataFrame:
    """9:30-9:35 bars (IEX feed) for each symbol and day in [start, end)."""
    from alpaca.data.enums import DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    key, secret = keys()
    client = StockHistoricalDataClient(key, secret)
    back = {to_alpaca(s): s for s in symbols}
    parts = []
    for i in range(0, len(symbols), 100):
        chunk = [to_alpaca(s) for s in symbols[i:i + 100]]
        for attempt in range(3):
            try:
                df = client.get_stock_bars(StockBarsRequest(
                    symbol_or_symbols=chunk, timeframe=TimeFrame(5, TimeFrameUnit.Minute),
                    start=start, end=end, feed=DataFeed.IEX)).df
                parts.append(df)
                break
            except Exception as exc:  # noqa: BLE001
                if "invalid symbol" in str(exc).lower():
                    for s in chunk:  # drop the bad ticker only
                        try:
                            parts.append(client.get_stock_bars(StockBarsRequest(
                                symbol_or_symbols=[s], timeframe=TimeFrame(5, TimeFrameUnit.Minute),
                                start=start, end=end, feed=DataFeed.IEX)).df)
                        except Exception:  # noqa: BLE001
                            pass
                    break
                log.warning("opening bars attempt %d failed: %s", attempt + 1, exc)
                time.sleep(5 * (attempt + 1))
    if not parts:
        return pd.DataFrame(columns=["symbol", "day", "open", "high", "low", "close", "volume"])
    bars = pd.concat(parts).reset_index()
    ts = pd.to_datetime(bars["timestamp"], utc=True).dt.tz_convert(NY)
    bars = bars[ts.dt.strftime("%H:%M") == "09:30"].copy()
    bars["day"] = ts[bars.index].dt.strftime("%Y-%m-%d")
    bars["symbol"] = bars["symbol"].map(lambda s: back.get(s, s))
    return bars[["symbol", "day", "open", "high", "low", "close", "volume"]]


# ---------------------------------------------------------------------------
# Selection and sizing (pure functions, unit-tested)
# ---------------------------------------------------------------------------
def select_stocks(today_bars: pd.DataFrame, baseline: pd.Series, stats: pd.DataFrame,
                  exclude: set[str]) -> list[dict]:
    """Rank today's opening bars by relative volume and return the top SIP_TOP picks."""
    picks = []
    for _, b in today_bars.iterrows():
        s = b["symbol"]
        if s in exclude or s not in stats.index or s not in baseline.index or baseline[s] <= 0:
            continue
        st = stats.loc[s]
        if st.prev_close < config.MIN_PRICE or st.avg_volume < C.MIN_AVG_VOLUME or st.atr < C.MIN_ATR:
            continue
        rvol = float(b["volume"]) / float(baseline[s])
        if rvol < C.MIN_RVOL:
            continue
        if b["close"] > b["open"]:
            side, level = "long", float(b["high"])
        elif b["close"] < b["open"] and True:
            side, level = "short", float(b["low"])
        else:
            continue
        picks.append({"symbol": s, "rvol": round(rvol, 2), "side": side, "level": round(level, 2),
                      "atr": round(float(st.atr), 4), "stop_dist": round(C.STOP_ATR * float(st.atr), 4),
                      "first_open": float(b["open"]), "first_close": float(b["close"])})
    picks.sort(key=lambda p: -p["rvol"])
    for i, p in enumerate(picks[:C.TOP], start=1):
        p["rank"] = i
    return picks[:C.TOP]


def size(equity: float, price: float, stop_dist: float, fractional: bool = True) -> float:
    """Shares to buy: RISK_PCT at the stop, at most MAX_POS_PCT of the account; fractional (4 decimals)
    unless the stock is not fractionable."""
    if stop_dist <= 0 or price <= 0:
        return 0
    q = max(0.0, min(equity * C.RISK_PCT / 100 / stop_dist, equity * C.MAX_POS_PCT / 100 / price))
    return math.floor(q * 10_000) / 10_000 if fractional else math.floor(q)


def fmt_qty(q: float) -> str:
    return str(int(q)) if q == int(q) else f"{q:g}"


# ---------------------------------------------------------------------------
# One management pass (unit-tested with a fake broker)
# ---------------------------------------------------------------------------
OPEN_STATUSES = {"new", "accepted", "pending_new", "partially_filled", "held", "accepted_for_bidding", "pending_replace"}


def manage(broker, picks: list[dict], day: str, t: datetime, since: datetime, allow_entries: bool, plan: dict,
           quotes: dict | None = None):
    """Place protective stops for filled entries, then enter picks that just broke out while there is room."""
    pre = prefix(day)
    orders = [o for o in broker.orders_today(since) if o["cid"].startswith(pre)]
    by_cid = {o["cid"]: o for o in orders}
    positions = broker.positions()
    acct = broker.account()
    equity = acct["equity"]

    # 1) protective stop for every filled (or partly filled) entry
    for o in orders:
        if "-e-" not in o["cid"] or o["filled_qty"] <= 0:
            continue
        sym = o["symbol"]
        pick = next((p for p in picks if p["symbol"] == sym), None)
        stop_cid = pre + "s-" + sym
        if stop_cid in by_cid or pick is None:
            continue
        held = positions.get(sym, {}).get("qty", 0.0)
        qty = round(min(abs(held), o["filled_qty"]) if held else o["filled_qty"], 9)
        if qty <= 0:
            continue
        fill = o["filled_avg_price"] or pick["level"]
        if pick["side"] == "long":
            stop_px, side = fill - pick["stop_dist"], "sell"
        else:
            stop_px, side = fill + pick["stop_dist"], "buy"
        broker.stop_order(sym, qty, side, stop_px, stop_cid)
        aimed = pick["level"]
        slip = (fill - aimed) / aimed * 100 if pick["side"] == "long" else (aimed - fill) / aimed * 100
        plan["events"].append({"time": hm(t), "symbol": sym, "event": "entry filled", "price": fill, "qty": qty,
                               "stop": round(stop_px, 2), "aimed": aimed, "slippage_pct": round(slip, 4)})
        log.info("Entry filled: %s %s x%s @ %.2f -> stop %.2f", pick["side"], sym, fmt_qty(qty), fill, stop_px)

    todo = [p for p in picks if not p.get("state")] if allow_entries else []
    # stops still at their first price, for the move to breakeven
    be_r = C.BREAKEVEN_R
    to_move = []
    if be_r:
        for o in orders:
            if "-s-" in o["cid"] and not o["cid"].endswith("-be") and o["status"] in OPEN_STATUSES                     and o["cid"] + "-be" not in by_cid:
                to_move.append(o)
    if not todo and not to_move:
        return
    if quotes is None:
        quotes = get_quotes(sorted({p["symbol"] for p in todo} | {o["symbol"] for o in to_move}), day)

    # 1b) once a trade is SIP_BREAKEVEN_R x its risk in profit, move its stop to the entry price
    for o in to_move:
        sym = o["symbol"]
        pick = next((p for p in picks if p["symbol"] == sym), None)
        entry = next((e for e in orders if "-e-" in e["cid"] and e["symbol"] == sym and e["filled_qty"] > 0), None)
        q = quotes.get(sym)
        if pick is None or entry is None or not q or not entry["filled_avg_price"]:
            continue
        fill, long = entry["filled_avg_price"], pick["side"] == "long"
        trigger = fill + be_r * pick["stop_dist"] if long else fill - be_r * pick["stop_dist"]
        if not ((q["high"] >= trigger) if long else (q["low"] <= trigger)):
            continue
        try:
            broker.move_stop(o["id"], fill, o["cid"] + "-be")
        except Exception as exc:  # noqa: BLE001  (e.g. the stop filled a moment ago)
            log.warning("Could not move %s's stop to breakeven: %s", sym, exc)
            continue
        plan["events"].append({"time": hm(t), "symbol": sym, "event": "stop moved to breakeven",
                               "price": q["last"], "stop": round(fill, 2)})
        log.info("Breakeven: %s %s reached %.2f (+%gR); stop moved to %.2f", pick["side"], sym, trigger, be_r, fill)

    if not todo:
        return

    # 2) entries: the first pick to break its level gets the room (like the backtest).
    #    Nothing rests at the broker, so several fills can never overshoot the limit.
    sip_syms = {o["symbol"] for o in orders if "-e-" in o["cid"]}
    invested = sum(abs(p["market_value"]) for s, p in positions.items() if s in sip_syms)
    # market entries sent but not filled yet still count against the limit
    pending = sum(p.get("notional", 0) for p in picks if p.get("state") == "entered"
                  and not any(o["cid"] == entry_cid(day, p) and o["filled_qty"] > 0 for o in orders))
    cap = equity * C.MAX_INVESTED_PCT / 100
    for p in todo:
        q = quotes.get(p["symbol"])
        if not q:
            continue
        long = p["side"] == "long"
        crossed = q["high"] >= p["level"] if long else q["low"] <= p["level"]
        if not crossed:
            continue
        px = q["last"]
        try:
            frac = broker.asset(p["symbol"]).get("fractionable", False)
        except Exception:  # noqa: BLE001
            frac = False
        qty = size(equity, px, p["stop_dist"], fractional=frac)
        notional = qty * px
        if qty <= 0 or notional < 1:
            p["state"] = "too small"
            plan["events"].append({"time": hm(t), "symbol": p["symbol"], "price": px,
                                   "event": "skipped: position too small" + ("" if frac else " (not fractionable)")})
            continue
        if invested + pending + notional > cap:
            p["state"] = "missed: no room"
            plan["events"].append({"time": hm(t), "symbol": p["symbol"], "event": "skipped: broke out while the 50% limit was full",
                                   "price": px})
            log.info("Breakout with no room: %s %s @ %.2f (limit full)", p["side"], p["symbol"], px)
            continue
        if not long:
            try:
                a = broker.asset(p["symbol"])
            except Exception:  # noqa: BLE001
                a = {"shortable": False, "easy_to_borrow": False}
            if not (a["shortable"] and a["easy_to_borrow"]):
                p["state"] = "not shortable"
                plan["events"].append({"time": hm(t), "symbol": p["symbol"], "event": "skipped: not shortable"})
                continue
        side = "buy" if long else "sell"
        broker.market_order(p["symbol"], qty, side, entry_cid(day, p))
        p.update(state="entered", detect_price=px, notional=notional, detect_time=hm(t))
        pending += notional
        plan["events"].append({"time": hm(t), "symbol": p["symbol"], "event": f"entry order {side} market",
                               "price": px, "aimed": p["level"], "qty": qty})
        log.info("Breakout: %s %s x%s at market (level %.2f, last %.2f, rank %d, rvol %.1f)", side, p["symbol"], fmt_qty(qty),
                 p["level"], px, p["rank"], p["rvol"])


def record_day(history: list[dict], plan: dict, equity_end: float) -> list[dict]:
    """Add (or replace) today's row: account value at the start and end of the day, and the trades."""
    events = plan.get("events", [])
    row = {"date": plan.get("date"), "equity_start": round(float(plan.get("equity_at_start") or 0), 2),
           "equity_end": round(float(equity_end), 2),
           "entries": sum(1 for e in events if str(e.get("event", "")).startswith("entry filled")),
           "stops_moved": sum(1 for e in events if e.get("event") == "stop moved to breakeven"),
           "dry_run": bool(plan.get("dry_run"))}
    row["pnl"] = round(row["equity_end"] - row["equity_start"], 2) if row["equity_start"] else None
    return [r for r in history if r.get("date") != row["date"]] + [row]


def long_only(picks: list[dict]) -> list[dict]:
    """A cash account cannot short: keep the up-moves, with their rank among all stocks in play."""
    return [p for p in picks if p["side"] == "long"]


def skip_late_breakouts(plan: dict, quotes: dict, t: datetime):
    """For a start after the open: picks already past their level broke out before the bot was
    running, so entering now would buy far from the breakout. Only breakouts from now on count."""
    for p in plan["picks"]:
        q = quotes.get(p["symbol"])
        if p.get("state") or not q:
            continue
        past = q["last"] >= p["level"] if p["side"] == "long" else q["last"] <= p["level"]
        if past:
            p["state"] = "missed: broke out before the bot started"
            plan["events"].append({"time": hm(t), "symbol": p["symbol"], "event": "skipped: broke out before the bot started",
                                   "price": q["last"]})
            log.info("Late start: %s already past its level (%.2f vs %.2f), skipped", p["symbol"], q["last"], p["level"])


def get_quotes(symbols: list[str], day: str) -> dict:
    """Latest trade plus the high/low of the current minute (IEX feed) for each symbol.
    Minute bars from before 9:35 are ignored: they belong to the first 5-minute candle."""
    from alpaca.data.enums import DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockSnapshotRequest

    global _data_client
    if _data_client is None:
        key, secret = keys()
        _data_client = StockHistoricalDataClient(key, secret)
    back = {to_alpaca(s): s for s in symbols}
    snaps = _data_client.get_stock_snapshot(StockSnapshotRequest(symbol_or_symbols=list(back), feed=DataFeed.IEX))
    start = datetime.strptime(f"{day} 09:35", "%Y-%m-%d %H:%M").replace(tzinfo=NY)
    out = {}
    for a_sym, sn in snaps.items():
        if not sn or not sn.latest_trade:
            continue
        last = float(sn.latest_trade.price)
        hi = lo = last
        mb = sn.minute_bar
        if mb is not None and pd.Timestamp(mb.timestamp).tz_convert(NY) >= pd.Timestamp(start):
            hi, lo = max(hi, float(mb.high)), min(lo, float(mb.low))
        out[back.get(a_sym, a_sym)] = {"last": last, "high": hi, "low": lo}
    return out


_data_client = None


def cancel_entries(broker, day, since, plan, t):
    pre = prefix(day)
    for o in broker.orders_today(since):
        if o["cid"].startswith(pre + "e-") and o["status"] in OPEN_STATUSES:
            broker.cancel(o["id"])
            plan["events"].append({"time": hm(t), "symbol": o["symbol"], "event": "entry cancelled (entry window over)"})
            log.info("Cancelled unfilled entry for %s", o["symbol"])


def flatten(broker, day, since, plan, t):
    """Cancel all of today's day-trade orders, then close the positions they opened."""
    pre = prefix(day)
    orders = [o for o in broker.orders_today(since) if o["cid"].startswith(pre)]
    for o in orders:
        if o["status"] in OPEN_STATUSES:
            broker.cancel(o["id"])
    time.sleep(0 if broker.__class__.__name__ != "Broker" else 3)
    positions = broker.positions()
    for sym in sorted({o["symbol"] for o in orders if "-e-" in o["cid"] and o["filled_qty"] > 0}):
        q = positions.get(sym, {}).get("qty", 0.0)
        if q == 0:
            continue
        side = "sell" if q > 0 else "buy"
        qty = abs(q) if abs(q) != int(abs(q)) else int(abs(q))
        broker.market_order(sym, qty, side, pre + "x-" + sym)
        plan["events"].append({"time": hm(t), "symbol": sym, "event": "closed at end of day", "qty": qty})
        log.info("Flatten: %s %s x%s", side, sym, fmt_qty(qty))


# ---------------------------------------------------------------------------
# Main phases
# ---------------------------------------------------------------------------
def keys():
    load_dotenv(BASE / ".env")
    return os.getenv("SMALL_ALPACA_API_KEY", "").strip(), os.getenv("SMALL_ALPACA_SECRET_KEY", "").strip()


def wait_until(target: datetime):
    while True:
        left = (target - now_ny()).total_seconds()
        if left <= 0:
            return
        time.sleep(min(left, 30))


def save_plan(plan: dict):
    (BASE / C.PLAN_FILE).write_text(json.dumps(plan, indent=1, default=str))


def load_plan(day: str) -> dict | None:
    p = BASE / C.PLAN_FILE
    if p.exists():
        try:
            d = json.loads(p.read_text())
            if d.get("date") == day:
                return d
        except json.JSONDecodeError:
            pass
    return None


def prepare(day: str):
    """Everything known before the open: S&P 500 list, daily stats, normal opening volume."""
    symbols = sp500_symbols()
    stats = daily_stats(symbols, day)
    d0 = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=NY)
    hist = opening_bars(list(stats.index), d0 - timedelta(days=30), d0)
    hist = hist[hist.day < day]
    baseline = hist.sort_values("day").groupby("symbol").tail(14).groupby("symbol")["volume"].mean()
    return stats, baseline


def build_picks(broker, day: str, exclude: set[str], prepared=None) -> list[dict]:
    stats, baseline = prepared or prepare(day)
    d0 = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=NY)
    for attempt in range(8):  # the 9:30-9:35 bars can take a few seconds to be published
        today = opening_bars(list(stats.index), d0 + timedelta(hours=9, minutes=30), d0 + timedelta(hours=9, minutes=36))
        today = today[today.day == day]
        if len(today) >= 100 or prepared is None:
            break
        log.info("Only %d opening bars so far; waiting for the rest", len(today))
        time.sleep(10)
    picks = select_stocks(today, baseline, stats, exclude)
    log.info("Stocks in play for %s: %s", day, ", ".join(f"{p['symbol']}({p['side'][0]},{p['rvol']}x)" for p in picks) or "none")
    return picks


def run(phase: str, dry_run: bool, past_day: str | None = None, test_minutes: int = 0):
    if not C.ENABLED and phase != "plan":
        log.info("Small bot is turned off (C.ENABLED = False).")
        return
    key, secret = keys()
    if not key or not secret:
        raise RuntimeError("SMALL_ALPACA_API_KEY / SMALL_ALPACA_SECRET_KEY are missing (the $1,000 account's keys)")
    broker = Broker(key, secret, dry_run)
    eq = broker.account()["equity"]
    if eq > C.MAX_ACCOUNT:
        log.error("This account is worth $%s, more than MAX_ACCOUNT ($%s): wrong keys? Not trading.",
                  f"{eq:,.0f}", f"{C.MAX_ACCOUNT:,.0f}")
        return
    if phase == "plan" and past_day:
        # Show what the bot would have picked on an earlier session (no waiting, no orders)
        picks = long_only(build_picks(broker, past_day, set()))
        equity = broker.account()["equity"]
        for p in picks:
            q = size(equity, p["level"], p["stop_dist"])
            log.info("  #%d %-5s %-5s rvol %5.1fx  entry stop %.2f  stop distance %.2f  -> %s shares (~$%s)",
                     p["rank"], p["symbol"], p["side"], p["rvol"], p["level"], p["stop_dist"], fmt_qty(q), f"{q * p['level']:,.0f}")
        return
    clock = broker.clock()
    t = now_ny()
    day = t.strftime("%Y-%m-%d")
    next_open = pd.Timestamp(clock.next_open).tz_convert(NY)
    if not clock.is_open and next_open.strftime("%Y-%m-%d") != day:
        if phase == "plan":
            log.info("Market closed today; nothing to plan.")
        else:
            log.info("Market is closed today. Nothing to do.")
        return
    close_t = pd.Timestamp(clock.next_close).tz_convert(NY).to_pydatetime()
    flat_at = min(datetime.strptime(f"{day} {C.FLATTEN_AT}", "%Y-%m-%d %H:%M").replace(tzinfo=NY),
                  close_t - timedelta(minutes=5))
    entry_until = min(datetime.strptime(f"{day} {C.ENTRY_UNTIL}", "%Y-%m-%d %H:%M").replace(tzinfo=NY),
                      flat_at - timedelta(minutes=10))
    job_until = min(datetime.strptime(f"{day} {C.DAY_JOB_UNTIL}", "%Y-%m-%d %H:%M").replace(tzinfo=NY),
                    flat_at - timedelta(minutes=5))
    since = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=NY)
    log.info("---- Day trader %s phase %s (dry_run=%s); entries until %s, flat at %s ----",
             day, phase, dry_run, hm(entry_until), hm(flat_at))

    plan = load_plan(day) or {"date": day, "picks": [], "events": [], "dry_run": dry_run}
    if phase in ("day", "plan") and not plan["picks"]:
        prepared = prepare(day)   # before the open, so picks are ready seconds after 9:35
        log.info("Prepared %d stocks before the open", len(prepared[0]))
        wait_until(datetime.strptime(f"{day} 09:35:05", "%Y-%m-%d %H:%M:%S").replace(tzinfo=NY))
        held = set(broker.positions())
        crossover_orders = {o["symbol"] for o in broker.orders_today(since) if not o["cid"].startswith("sml-")}
        plan["picks"] = long_only(build_picks(broker, day, held | crossover_orders, prepared))
        plan["equity_at_start"] = broker.account()["equity"]
        if phase == "day" and now_ny() > datetime.strptime(f"{day} 09:40", "%Y-%m-%d %H:%M").replace(tzinfo=NY):
            skip_late_breakouts(plan, get_quotes([p["symbol"] for p in plan["picks"]], day), now_ny())
        save_plan(plan)
        if phase == "plan":
            return

    end = job_until if phase == "day" else flat_at
    if test_minutes:  # short test runs: stop after a few minutes
        end = min(end, now_ny() + timedelta(minutes=test_minutes))
    last_save = 0.0
    while now_ny() < end:
        t = now_ny()
        entries_open = t < entry_until
        try:
            manage(broker, plan["picks"], day, t, since, allow_entries=entries_open, plan=plan)
        except Exception as exc:  # noqa: BLE001
            log.warning("Management pass failed: %s", exc)
        if time.time() - last_save > 30:
            save_plan(plan)
            last_save = time.time()
        time.sleep(FAST_POLL_SECONDS if entries_open else POLL_SECONDS)

    t = now_ny()
    if phase == "day":
        manage(broker, plan["picks"], day, t, since, allow_entries=False, plan=plan)
    else:
        for p in plan["picks"]:
            if not p.get("state"):
                p["state"] = "no breakout"
        flatten(broker, day, since, plan, t)
        time.sleep(15)
        left = {s: p for s, p in broker.positions().items()
                if s in {o["symbol"] for o in broker.orders_today(since) if o["cid"].startswith(prefix(day) + "e-")}}
        log.info("End of day: %s", "all day-trade positions closed" if not left else f"STILL OPEN: {left}")
        try:
            hp = BASE / C.HISTORY_FILE
            history = json.loads(hp.read_text()) if hp.exists() else []
            eq_end = broker.account()["equity"]
            hp.write_text(json.dumps(record_day(history, plan, eq_end), indent=1))
            log.info("Account at the close: $%.2f (started the day at $%.2f)", eq_end, plan.get("equity_at_start") or 0)
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not save the day's history: %s", exc)
    save_plan(plan)


def main():
    ap = argparse.ArgumentParser(description="Stocks in Play day trader (paper)")
    ap.add_argument("--phase", choices=["day", "close", "plan"], required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--day", help="with --phase plan: show the picks for an earlier session (YYYY-MM-DD)")
    ap.add_argument("--minutes", type=int, default=0, help="stop the phase after this many minutes (testing)")
    a = ap.parse_args()
    if a.minutes and not (a.dry_run or C.DRY_RUN):
        ap.error("--minutes is only for dry runs")
    try:
        run(a.phase, a.dry_run or C.DRY_RUN, a.day, a.minutes)
    except Exception as exc:  # noqa: BLE001
        log.exception("Day trader failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
