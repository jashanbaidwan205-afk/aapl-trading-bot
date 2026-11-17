"""
Backtest of an opening range breakout (ORB) day-trading strategy on the
same most-active stock list the live bot uses.

Rules (all positions are closed the same day):
  * Stock list: each day, S&P 500 stocks ranked by the previous day's share
    volume, top MOST_ACTIVE_TOP kept (as Alpaca's most-active list), skipping
    EXCLUDE_SYMBOLS and stocks that closed under MIN_PRICE.
  * Opening range: high and low of the first OR_MINUTES after 9:30 AM ET.
  * Entry (long): the first 5-minute bar that CLOSES above the range high,
    between the end of the range and ENTRY_UNTIL. Fill at the next bar's open.
    (Short version: a close below the range low, mirrored.)
  * Stop: the other side of the range. Target: TARGET_R x the risk per share.
    If a bar touches both, the stop is assumed to hit first.
  * Exit at the 3:55 PM bar's close if neither stop nor target hit.
  * Size: risk RISK_PCT of the account per trade, at most MAX_POS_PCT of the
    account per position, at most MAX_TRADES_PER_DAY trades a day, and open
    positions never worth more than MAX_INVESTED_PCT of the account.
  * SLIPPAGE_PCT against you on every fill.
  * Optional: at each month end, withdraw a share of the month's net profit.

Data: 5-minute bars from Alpaca (needs ALPACA_API_KEY / ALPACA_SECRET_KEY),
daily prices from Yahoo Finance. Limits: today's S&P 500 members only; 5-minute
bars can't show the order of moves inside a bar.

Usage: python backtest_orb.py [--years 1]
Writes backtest_orb_results/<variant>/{summary,equity,trades,months}.json
"""

import argparse
import io
import json
import math
import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv

import config

BASE = Path(__file__).resolve().parent
OUT = BASE / "backtest_orb_results"
CACHE_FILE = BASE / "backtest_orb_cache" / "bars_5m_v2.pkl"   # saved between runs (with volume + VWAP)
BAR_COLS = ["day", "hm", "open", "high", "low", "close", "volume", "vwap"]
NY = "America/New_York"
START_CASH = 100_000.0

P = {
    "OR_MINUTES": 15,
    "ENTRY_UNTIL": "11:30",
    "TARGET_R": 2.0,
    "RISK_PCT": 0.5,
    "MAX_POS_PCT": 10.0,
    "MAX_TRADES_PER_DAY": 5,
    "MAX_INVESTED_PCT": getattr(config, "MAX_INVESTED_PCT", 50.0),
    "SLIPPAGE_PCT": 0.05,
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def sp500_symbols() -> list[str]:
    html = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                        headers={"User-Agent": "Mozilla/5.0 (backtest script)"}, timeout=30).text
    table = pd.read_html(io.StringIO(html))[0]
    return sorted({str(s).replace(".", "-") for s in table["Symbol"]})


def daily_data(symbols: list[str], years: int) -> dict[str, pd.DataFrame]:
    frames = []
    for i in range(0, len(symbols), 100):
        chunk = symbols[i:i + 100]
        for attempt in range(3):
            df = yf.download(chunk, period=f"{years + 1}y", interval="1d", auto_adjust=True,
                             group_by="ticker", threads=True, progress=False)
            if not df.empty:
                frames.append(df)
                break
            time.sleep(5 * (attempt + 1))
    raw = pd.concat(frames, axis=1)
    out = {f: raw.xs(f, axis=1, level=1).sort_index() for f in ("Open", "High", "Low", "Close", "Volume")}
    return {f: d.loc[:, ~d.columns.duplicated()] for f, d in out.items()}


def minute_bars(symbols: list[str], start, end) -> pd.DataFrame:
    """5-minute regular-hours bars from Alpaca, indexed by (symbol, NY timestamp)."""
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    load_dotenv(BASE / ".env")
    key, secret = os.getenv("ALPACA_API_KEY", ""), os.getenv("ALPACA_SECRET_KEY", "")
    if not key or not secret:
        raise SystemExit("ALPACA_API_KEY / ALPACA_SECRET_KEY are needed for minute data")
    client = StockHistoricalDataClient(key, secret)
    feed = DataFeed.SIP
    to_alpaca = {s: s.replace("-", ".") for s in symbols}  # Yahoo BRK-B -> Alpaca BRK.B
    from_alpaca = {v: k for k, v in to_alpaca.items()}
    start_day, end_day = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

    # Saved bars from earlier runs: {"bars": {sym: DataFrame}, "span": {sym: (first_day, last_day)}}
    cache = {"bars": {}, "span": {}}
    if CACHE_FILE.exists():
        try:
            cache = pickle.loads(CACHE_FILE.read_bytes())
        except Exception:  # noqa: BLE001
            print("  saved bars unreadable; downloading fresh")

    # Work out what each stock still needs: nothing, the newest days, or everything
    jobs: dict[str, list[str]] = {}
    for s in symbols:
        first, last = cache["span"].get(s, (None, None))
        if first is None or first > start_day:
            jobs.setdefault(start_day, []).append(s)
        elif last < end_day:
            nxt = (pd.Timestamp(last) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            jobs.setdefault(nxt, []).append(s)
    todo = sum(len(v) for v in jobs.values())
    print(f"  saved bars cover {len(symbols) - todo}/{len(symbols)} stocks; downloading {todo}")

    def fetch(chunk, from_day):
        nonlocal feed
        t0 = pd.Timestamp(from_day, tz=NY).to_pydatetime()
        for attempt in range(4):
            try:
                req = StockBarsRequest(symbol_or_symbols=[to_alpaca[s] for s in chunk],
                                       timeframe=TimeFrame(5, TimeFrameUnit.Minute),
                                       start=t0, end=end, adjustment=Adjustment.SPLIT, feed=feed)
                return client.get_stock_bars(req).df
            except Exception as exc:  # noqa: BLE001
                msg = str(exc).lower()
                if feed == DataFeed.SIP and "subscription" in msg:
                    print("  SIP data not allowed on this plan; using IEX data instead")
                    feed = DataFeed.IEX
                    continue
                if "invalid symbol" in msg:
                    raise ValueError(msg) from exc
                print(f"  minute bars attempt {attempt + 1} failed: {exc}")
                time.sleep(10 * (attempt + 1))
        return None

    def fetch_chunk(chunk, from_day):
        """Returns (frames, missing). One bad ticker only drops that stock."""
        try:
            df = fetch(chunk, from_day)
            return ([df] if df is not None else []), ([] if df is not None else list(chunk))
        except ValueError:
            frames, missing = [], []
            for s in chunk:
                try:
                    df = fetch([s], from_day)
                    (frames.append(df) if df is not None else missing.append(s))
                except ValueError:
                    missing.append(s)
            return frames, missing

    tasks = [(syms[i:i + 25], day) for day, syms in jobs.items() for i in range(0, len(syms), 25)]
    parts, missing, done = [], [], 0
    with ThreadPoolExecutor(max_workers=3) as pool:  # 3 at once stays inside Alpaca's free rate limit
        for frames, miss in pool.map(lambda t: fetch_chunk(*t), tasks):
            parts += frames
            missing += miss
            done += 1
            print(f"  downloaded batch {done}/{len(tasks)}")
    if missing:
        print(f"  no minute data for {len(missing)} stock(s): {', '.join(missing)}")

    if parts:
        new = pd.concat(parts).reset_index()
        new["symbol"] = new["symbol"].map(lambda s: from_alpaca.get(s, s))
        ts = pd.to_datetime(new["timestamp"], utc=True).dt.tz_convert(NY)
        new["day"], new["hm"] = ts.dt.strftime("%Y-%m-%d"), ts.dt.strftime("%H:%M")
        new = new[(new.hm >= "09:30") & (new.hm < "16:00")][["symbol"] + BAR_COLS]
        fetched = {s for day_syms in jobs.values() for s in day_syms} - set(missing)
        for s, g in new.groupby("symbol"):
            old = cache["bars"].get(s)
            g = g.drop(columns="symbol")
            if old is not None and cache["span"][s][0] <= start_day:
                old = old.astype({"day": str, "hm": str})
                g = pd.concat([old, g]).drop_duplicates(["day", "hm"], keep="last")
            g = g.sort_values(["day", "hm"]).reset_index(drop=True)
            # Categories keep memory and the saved file small (dates/times repeat a lot)
            cache["bars"][s] = g.astype({"day": "category", "hm": "category", "volume": "float32", "vwap": "float32"})
        for s in fetched:
            first = cache["span"].get(s, (start_day, None))[0]
            cache["span"][s] = (min(first, start_day) if first else start_day, end_day)
        CACHE_FILE.parent.mkdir(exist_ok=True)
        CACHE_FILE.write_bytes(pickle.dumps(cache))

    want, out = set(symbols), {}
    for s, g in cache["bars"].items():
        if s in want:
            d = g["day"].astype(str)
            out[s] = g[(d >= start_day) & (d <= end_day)]
    print(f"  {sum(len(g) for g in out.values()):,} bars for {len(out)} stocks ({feed.value} feed)")
    return out


def day_bars(bars: dict[str, pd.DataFrame]) -> dict:
    """{(symbol, day): DayBars} from minute_bars() output."""
    return {(s, str(day)): DayBars(g) for s, df in bars.items() for day, g in df.groupby("day", observed=True)}


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
def or_end_hm() -> str:
    m = 30 + P["OR_MINUTES"]
    return f"{9 + m // 60:02d}:{m % 60:02d}"


class DayBars:
    """One stock's bars for one day as NumPy arrays (much faster than DataFrame rows)."""
    __slots__ = ("hm", "open", "high", "low", "close", "volume", "vwap")

    def __init__(self, df: pd.DataFrame):
        df = df.sort_values("hm")
        self.hm = df["hm"].astype(str).to_numpy()
        self.open, self.high = df["open"].to_numpy(float), df["high"].to_numpy(float)
        self.low, self.close = df["low"].to_numpy(float), df["close"].to_numpy(float)
        self.volume = df["volume"].to_numpy(float) if "volume" in df else np.zeros(len(df))
        self.vwap = df["vwap"].to_numpy(float) if "vwap" in df else self.close

    def __len__(self):
        return len(self.hm)


_found: dict = {}


def find_trade(b: DayBars, side: str, key=None):
    """Return (entry_hm, entry_px, stop, entry_index) for the day's first breakout, or None."""
    if key is not None:
        if (key, side) not in _found:
            _found[(key, side)] = find_trade(b, side)
        return _found[(key, side)]
    end_or = or_end_hm()
    in_range = b.hm < end_or
    if in_range.sum() < P["OR_MINUTES"] // 5:
        return None
    hi, lo = b.high[in_range].max(), b.low[in_range].min()
    window = (~in_range) & (b.hm < P["ENTRY_UNTIL"])
    trig = window & ((b.close > hi) if side == "long" else (b.close < lo))
    trig[-1] = False  # need a next bar to enter on
    if not trig.any():
        return None
    k = int(np.argmax(trig)) + 1
    return b.hm[k], float(b.open[k]), float(lo if side == "long" else hi), k


def exit_trade(b: DayBars, k: int, side: str, entry: float, stop: float, target: float):
    """Exit from bar k on: stop first, then target, else the last bar's close."""
    slip = P["SLIPPAGE_PCT"] / 100
    lo, hi, op = b.low[k:], b.high[k:], b.open[k:]
    hit_stop = (lo <= stop) if side == "long" else (hi >= stop)
    hit_tgt = (hi >= target) if side == "long" else (lo <= target)
    n = len(lo)
    i_stop = int(np.argmax(hit_stop)) if hit_stop.any() else n
    i_tgt = int(np.argmax(hit_tgt)) if hit_tgt.any() else n
    if i_stop <= i_tgt and i_stop < n:  # same bar: assume the stop hit first
        px = min(stop, op[i_stop]) if side == "long" else max(stop, op[i_stop])
        return b.hm[k + i_stop], px * ((1 - slip) if side == "long" else (1 + slip)), "stop"
    if i_tgt < n:
        px = max(target, op[i_tgt]) if side == "long" else min(target, op[i_tgt])
        return b.hm[k + i_tgt], px * ((1 - slip) if side == "long" else (1 + slip)), "target"
    return b.hm[-1], float(b.close[-1]) * ((1 - slip) if side == "long" else (1 + slip)), "close"


def simulate(days, lists, bars_by, spy, sides, withdraw_pct):
    slip = P["SLIPPAGE_PCT"] / 100
    cash = START_CASH
    trades, equity_rows, months = [], [], []
    month_pnl = withdrawn = 0.0
    for di, day in enumerate(days):
        equity = cash
        cands = []
        for rank, sym in enumerate(lists[day], start=1):
            b = bars_by.get((sym, day))
            if b is None or len(b) == 0:
                continue
            for side in sides:
                ft = find_trade(b, side, key=(sym, day))
                if ft:
                    cands.append((ft[0], rank, sym, side, ft))
        cands.sort(key=lambda c: (c[0], c[1]))
        open_trades, taken = [], 0
        day_pnl = 0.0
        for hm, rank, sym, side, (ehm, raw_entry, stop, k) in cands:
            if taken >= P["MAX_TRADES_PER_DAY"]:
                break
            open_trades = [t for t in open_trades if t["exit_hm"] > ehm]
            if any(t["symbol"] == sym for t in open_trades):
                continue
            entry = raw_entry * ((1 + slip) if side == "long" else (1 - slip))
            risk_ps = (entry - stop) if side == "long" else (stop - entry)
            if risk_ps <= 0:
                continue
            qty = math.floor(min(equity * P["RISK_PCT"] / 100 / risk_ps, equity * P["MAX_POS_PCT"] / 100 / entry))
            if qty < 1:
                continue
            invested = sum(t["entry"] * t["shares"] for t in open_trades)
            if invested + qty * entry > equity * P["MAX_INVESTED_PCT"] / 100:
                continue
            target = entry + P["TARGET_R"] * risk_ps if side == "long" else entry - P["TARGET_R"] * risk_ps
            xhm, xpx, why = exit_trade(bars_by[(sym, day)], k, side, entry, stop, target)
            pnl = (xpx - entry) * qty if side == "long" else (entry - xpx) * qty
            t = {"day": day, "symbol": sym, "side": side, "entry_hm": ehm, "exit_hm": xhm, "shares": qty,
                 "entry": round(entry, 2), "stop": round(stop, 2), "target": round(target, 2),
                 "exit": round(xpx, 2), "exit_reason": why, "pnl": round(pnl, 2),
                 "r_multiple": round(pnl / (risk_ps * qty), 2)}
            trades.append(t)
            open_trades.append(t)
            taken += 1
            day_pnl += pnl
        cash += day_pnl
        month_pnl += day_pnl
        month_end = di == len(days) - 1 or days[di + 1][:7] != day[:7]
        if month_end:
            take = max(0.0, month_pnl) * withdraw_pct / 100
            cash -= take
            withdrawn += take
            months.append({"month": day[:7], "realized_pl": round(month_pnl, 2), "withdrawn": round(take, 2),
                           "withdrawn_total": round(withdrawn, 2), "account_value": round(cash, 2)})
            month_pnl = 0.0
        equity_rows.append({"date": day, "equity": round(cash, 2), "trades": taken,
                            "withdrawn_total": round(withdrawn, 2), "spy_equity": round(spy[day], 2)})
    return trades, equity_rows, months, withdrawn


def summarize(trades, equity_rows, months, withdrawn, sides, withdraw_pct, feed_note):
    eq = pd.Series([r["equity"] + r["withdrawn_total"] for r in equity_rows])
    acct = pd.Series([r["equity"] for r in equity_rows])
    spy = pd.Series([r["spy_equity"] for r in equity_rows])
    dd = lambda s: round(float((s / s.cummax() - 1).min() * 100), 2)  # noqa: E731
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    gross_win, gross_loss = sum(t["pnl"] for t in wins), -sum(t["pnl"] for t in losses)
    end = equity_rows[-1]["equity"]
    return {
        "start": equity_rows[0]["date"], "end": equity_rows[-1]["date"], "trading_days": len(equity_rows),
        "sides": "+".join(sides), "withdraw_pct": withdraw_pct,
        "start_value": START_CASH, "end_value": round(end, 2), "total_withdrawn": round(withdrawn, 2),
        "end_value_plus_withdrawn": round(end + withdrawn, 2),
        "total_return_incl_withdrawn_pct": round((end + withdrawn) / START_CASH * 100 - 100, 2),
        "max_drawdown_pct": dd(acct), "max_drawdown_incl_withdrawn_pct": dd(eq),
        "spy_end_value": round(float(spy.iloc[-1]), 2),
        "spy_total_return_pct": round(float(spy.iloc[-1]) / START_CASH * 100 - 100, 2),
        "spy_max_drawdown_pct": dd(spy),
        "trades": len(trades), "wins": len(wins), "losses": len(losses),
        "win_rate_pct": round(len(wins) / len(trades) * 100, 1) if trades else None,
        "avg_win": round(gross_win / len(wins), 2) if wins else None,
        "avg_loss": round(-gross_loss / len(losses), 2) if losses else None,
        "profit_factor": round(gross_win / gross_loss, 2) if gross_loss else None,
        "avg_r": round(sum(t["r_multiple"] for t in trades) / len(trades), 3) if trades else None,
        "net_pl": round(sum(t["pnl"] for t in trades), 2),
        "avg_trades_per_day": round(len(trades) / len(equity_rows), 2),
        "exits_stop": sum(t["exit_reason"] == "stop" for t in trades),
        "exits_target": sum(t["exit_reason"] == "target" for t in trades),
        "exits_close": sum(t["exit_reason"] == "close" for t in trades),
        "months_positive": sum(m["realized_pl"] > 0 for m in months), "months": len(months),
        "best_month_pl": max(m["realized_pl"] for m in months), "worst_month_pl": min(m["realized_pl"] for m in months),
        "settings": (f"top {config.MOST_ACTIVE_TOP} by volume, min ${config.MIN_PRICE:g}, {P['OR_MINUTES']}-min range, "
                     f"entries until {P['ENTRY_UNTIL']}, target {P['TARGET_R']:g}R, risk {P['RISK_PCT']:g}% per trade, "
                     f"max {P['MAX_POS_PCT']:g}% per position, max {P['MAX_TRADES_PER_DAY']} trades/day, "
                     f"max {P['MAX_INVESTED_PCT']:g}% invested, {P['SLIPPAGE_PCT']:g}% slippage"),
        "data": feed_note,
        "run_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Opening range breakout backtest")
    ap.add_argument("--years", type=int, default=1)
    a = ap.parse_args()

    symbols = sp500_symbols()
    print(f"S&P 500 list: {len(symbols)} stocks")
    daily = daily_data(symbols + ["SPY"], a.years)
    closes, volumes, opens = daily["Close"], daily["Volume"], daily["Open"]

    # Test window: last N trading days that are fully in the past
    today = pd.Timestamp.now(tz=NY).strftime("%Y-%m-%d")
    idx = [d for d in closes.index if d.strftime("%Y-%m-%d") < today]
    window = idx[-252 * a.years:]
    excluded = {s.upper() for s in config.EXCLUDE_SYMBOLS} | {"SPY"}
    lists = {}
    for d in window:
        i = closes.index.get_loc(d)
        vol = volumes.iloc[i - 1].drop(labels=[c for c in excluded if c in volumes.columns], errors="ignore").dropna()
        ok = [s for s in vol.sort_values(ascending=False).index if closes.iloc[i - 1][s] >= config.MIN_PRICE]
        lists[d.strftime("%Y-%m-%d")] = ok[:config.MOST_ACTIVE_TOP]
    days = list(lists)
    union = sorted({s for v in lists.values() for s in v})
    print(f"Test window {days[0]} to {days[-1]}: {len(days)} days, {len(union)} different stocks")

    start = pd.Timestamp(days[0], tz=NY)
    end = pd.Timestamp(days[-1], tz=NY) + pd.Timedelta(hours=23)
    bars = minute_bars(union, start.to_pydatetime(), end.to_pydatetime())
    feed_note = "Alpaca 5-minute bars (split-adjusted); Yahoo Finance daily prices for the stock list and SPY"
    bars_by = day_bars(bars)

    spy_shares = START_CASH / float(opens["SPY"].loc[window[0]])
    spy = {d.strftime("%Y-%m-%d"): spy_shares * float(closes["SPY"].loc[d]) for d in window}

    meta = [{"label": "backtest_orb.py", "value": Path(__file__).read_text(encoding="utf-8"), "type": "code", "lang": "python"},
            {"label": "config.py", "value": (BASE / "config.py").read_text(encoding="utf-8"), "type": "code", "lang": "python"},
            {"label": "Input", "value": feed_note}]
    variants = {"long": (["long"], 0.0), "long_withdraw70": (["long"], 70.0), "long_short": (["long", "short"], 0.0),
                "long_short_withdraw70": (["long", "short"], 70.0)}
    for name, (sides, wd) in variants.items():
        trades, eq, months, withdrawn = simulate(days, lists, bars_by, spy, sides, wd)
        s = summarize(trades, eq, months, withdrawn, sides, wd, feed_note)
        out = OUT / name
        out.mkdir(parents=True, exist_ok=True)
        note = [{"label": "Made by", "value": "backtest_orb.py (code attached to the summary source)"}]
        (out / "summary.json").write_text(json.dumps({"rows": [s], "meta": meta}, indent=1))
        (out / "equity.json").write_text(json.dumps({"rows": eq, "meta": note}, separators=(",", ":")))
        (out / "trades.json").write_text(json.dumps({"rows": sorted(trades, key=lambda t: (t["day"], t["entry_hm"]), reverse=True), "meta": note}, separators=(",", ":")))
        (out / "months.json").write_text(json.dumps({"rows": months, "meta": note}, indent=1))
        print(f"\n== {name} ==")
        print(json.dumps({k: s[k] for k in ("end_value", "total_withdrawn", "end_value_plus_withdrawn",
                                             "total_return_incl_withdrawn_pct", "max_drawdown_pct", "trades",
                                             "win_rate_pct", "avg_win", "avg_loss", "profit_factor", "avg_r",
                                             "months_positive", "months", "spy_total_return_pct")}, indent=1))


if __name__ == "__main__":
    main()
