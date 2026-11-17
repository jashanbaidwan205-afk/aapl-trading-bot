"""
Backtests of three well-known professional day-trading methods on S&P 500 stocks.

1. "Stocks in Play" 5-minute opening range breakout (Zarattini & Aziz, 2023)
   - Each day keep the TOP_SIP stocks with the highest opening relative volume
     (first 5-minute volume / its average over the previous 14 days, at least 1x),
     price >= MIN_PRICE, 14-day average volume >= 1M shares, 14-day ATR >= $0.50.
   - Direction from the first 5-minute candle: up candle -> buy a break of its high,
     down candle -> sell short a break of its low (no trade on a flat candle).
   - Stop 10% of the 14-day ATR from entry. No profit target: exit at the close.

2. VWAP pullback (trend continuation used by institutional desks)
   - Same "stocks in play" list. Between 10:00 and 14:00, in an uptrend (price above a
     rising VWAP and above the day's open) buy the first bar that dips to VWAP and
     closes back above it; mirror for shorts in a downtrend.
   - Enter at the next bar's open. Stop just under the pullback bar's low
     (2% of ATR below). Target 2x the risk, else exit at the close.

3. Gap and go (momentum method popular with retail day traders)
   - Stocks that open >= 3% above (or below) yesterday's close with opening relative
     volume >= 2x; the TOP_GAP biggest gaps.
   - Gap up: buy a break of the first 5-minute candle's high before 10:30, stop at its
     low. Gap down: mirror. Target 2x the risk, else exit at the close.

Shared rules (same limits as the live bot): risk RISK_PCT of the account per trade,
at most MAX_POS_PCT of the account per position, positions together worth at most
MAX_INVESTED_PCT of the account (no margin), SLIPPAGE_PCT against you on every fill,
everything closed by 4 PM. Inside a 5-minute bar, prices are assumed to move
open -> low -> high -> close on a rising bar and open -> high -> low -> close on a
falling bar (the usual backtesting convention); stops and targets fill at their
level, or at the open if a bar opens past them. Each strategy also runs with 70% of
each month's net profit withdrawn.

Data: Alpaca 5-minute bars (split-adjusted, with volume and VWAP), Yahoo Finance daily
prices. Limits: today's S&P 500 members only; 5-minute bars can't show the order of
moves inside a bar; one year of history.

Usage: python backtest_daytrade.py [--years 1]
Writes backtest_daytrade_results/dt_{summary,equity,months,trades}.json
"""

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import backtest_orb as orb
import config

BASE = Path(__file__).resolve().parent


def to_json(obj, **kw) -> str:
    """json.dumps that also handles NumPy numbers."""
    return json.dumps(obj, default=lambda o: o.item() if hasattr(o, "item") else str(o), **kw)
OUT = BASE / "backtest_daytrade_results"
START_CASH = orb.START_CASH

S = {
    "RISK_PCT": 0.5,
    "MAX_POS_PCT": 10.0,
    "MAX_INVESTED_PCT": getattr(config, "MAX_INVESTED_PCT", 50.0),
    "SLIPPAGE_PCT": 0.05,
    "TOP_SIP": 20,
    "TOP_GAP": 10,
    "MIN_RVOL_SIP": 1.0,
    "MIN_RVOL_GAP": 2.0,
    "MIN_GAP_PCT": 3.0,
    "SIP_STOP_ATR": 0.10,
    "VWAP_STOP_ATR": 0.02,
    "TARGET_R": 2.0,
}

STRATEGIES = {
    "sip": ("Stocks in Play 5-min breakout", 20),
    "vwap": ("VWAP pullback", 10),
    "gap": ("Gap and go", 10),
    "sip_live": ("Day trader bot (its exact live rules)", 20),
}
LIVE_ENTRY_UNTIL = "15:00"   # sip_bot.py: SIP_ENTRY_UNTIL
LIVE_FLATTEN_AT = "15:55"    # sip_bot.py: SIP_FLATTEN_AT


# ---------------------------------------------------------------------------
# Candidate trades for one stock-day. Each returns a dict or None:
#   {"side", "k" (entry bar index), "entry_raw", "stop", "target" (None = hold to close),
#    "same_bar_target" (may the target fill on the entry bar)}
# ---------------------------------------------------------------------------
def first_index(mask: np.ndarray) -> int | None:
    return int(np.argmax(mask)) if mask.any() else None


def sip_trade(b, atr):
    if b.hm[0] != "09:30" or len(b) < 3:
        return None
    o, c = b.open[0], b.close[0]
    if c > o:
        side, level = "long", b.high[0]
        hit = b.high[1:] >= level
    elif c < o:
        side, level = "short", b.low[0]
        hit = b.low[1:] <= level
    else:
        return None
    hit &= b.hm[1:] < "15:30"
    j = first_index(hit)
    if j is None:
        return None
    k = j + 1
    entry = max(level, b.open[k]) if side == "long" else min(level, b.open[k])
    dist = S["SIP_STOP_ATR"] * atr
    stop = entry - dist if side == "long" else entry + dist
    return {"side": side, "k": k, "entry_raw": float(entry), "stop": float(stop), "target": None, "same_bar_target": False}


def vwap_trade(b, atr):
    if b.hm[0] != "09:30" or len(b) < 10:
        return None
    vol = np.where(b.volume > 0, b.volume, 1.0)
    cv = np.cumsum(b.vwap * vol) / np.cumsum(vol)
    day_open = b.open[0]
    for i in range(6, len(b) - 1):
        if b.hm[i] < "10:00":
            continue
        if b.hm[i] >= "14:00":
            break
        up = (b.close[i - 1] > cv[i - 1] and b.low[i] <= cv[i] * 1.001 and b.close[i] > cv[i]
              and cv[i] > cv[i - 6] and b.close[i] > day_open)
        dn = (b.close[i - 1] < cv[i - 1] and b.high[i] >= cv[i] * 0.999 and b.close[i] < cv[i]
              and cv[i] < cv[i - 6] and b.close[i] < day_open)
        if up or dn:
            side = "long" if up else "short"
            k = i + 1
            entry = b.open[k]
            stop = b.low[i] - S["VWAP_STOP_ATR"] * atr if up else b.high[i] + S["VWAP_STOP_ATR"] * atr
            risk = entry - stop if up else stop - entry
            if risk <= 0:
                return None
            target = entry + S["TARGET_R"] * risk if up else entry - S["TARGET_R"] * risk
            return {"side": side, "k": k, "entry_raw": float(entry), "stop": float(stop), "target": float(target),
                    "same_bar_target": True}
    return None


def gap_trade(b, gap_up: bool):
    if b.hm[0] != "09:30" or len(b) < 3:
        return None
    if gap_up:
        side, level, stop = "long", b.high[0], b.low[0]
        hit = b.high[1:] >= level
    else:
        side, level, stop = "short", b.low[0], b.high[0]
        hit = b.low[1:] <= level
    hit &= b.hm[1:] < "10:30"
    j = first_index(hit)
    if j is None:
        return None
    k = j + 1
    entry = max(level, b.open[k]) if gap_up else min(level, b.open[k])
    risk = entry - stop if gap_up else stop - entry
    if risk <= 0:
        return None
    target = entry + S["TARGET_R"] * risk if gap_up else entry - S["TARGET_R"] * risk
    return {"side": side, "k": k, "entry_raw": float(entry), "stop": float(stop), "target": float(target),
            "same_bar_target": False}


def bar_path(o, h, l, c):
    """Standard assumption for the order of prices inside a bar:
    a rising bar goes open -> low -> high -> close, a falling bar open -> high -> low -> close."""
    return [o, l, h, c] if c >= o else [o, h, l, c]


def exit_from(b, k, side, stop, target, entry_px):
    """Follow each bar's price path from the entry on; exit at the first stop or target
    crossed, else at the last bar's close. All exits pay slippage."""
    slip = S["SLIPPAGE_PCT"] / 100
    long = side == "long"
    out = (lambda px: px * (1 - slip)) if long else (lambda px: px * (1 + slip))
    for i in range(k, len(b)):
        path = bar_path(b.open[i], b.high[i], b.low[i], b.close[i])
        if i == k:
            # Skip the part of the entry bar before the entry price was first reached
            j = 0
            for j in range(1, 4):
                a, z = path[j - 1], path[j]
                if min(a, z) <= entry_px <= max(a, z):
                    break
            path = [entry_px] + path[j:]
        prev = path[0]
        if i > k:  # the bar opened past a level: fill at the open
            if (long and prev <= stop) or (not long and prev >= stop):
                return b.hm[i], out(prev), "stop"
            if target is not None and ((long and prev >= target) or (not long and prev <= target)):
                return b.hm[i], out(prev), "target"
        for p in path[1:]:
            lo, hi = min(prev, p), max(prev, p)
            # whichever level the move from prev to p reaches first
            hits = []
            if lo <= stop <= hi:
                hits.append((abs(stop - prev), stop, "stop"))
            if target is not None and lo <= target <= hi:
                hits.append((abs(target - prev), target, "target"))
            if hits:
                _, px, why = min(hits)
                return b.hm[i], out(px), why
            prev = p
    return b.hm[-1], out(b.close[-1]), "close"


# ---------------------------------------------------------------------------
# Daily selection lists and simulation
# ---------------------------------------------------------------------------
def build_candidates(days, bars_by, daily, symbols):
    """For each strategy and day: list of (entry_hm, rank, symbol, trade dict)."""
    closes, highs, lows, vols = daily["Close"], daily["High"], daily["Low"], daily["Volume"]
    prev_c = closes.shift(1)
    tr = pd.concat([highs - lows, (highs - prev_c).abs(), (lows - prev_c).abs()]).groupby(level=0).max()
    atr = tr.rolling(14).mean().shift(1)          # known before today's open
    avg_vol = vols.rolling(14).mean().shift(1)

    # Opening (first 5-minute) volume per stock per day, and its 14-day average before today
    first_vol = {}
    for (s, d), b in bars_by.items():
        if len(b) and b.hm[0] == "09:30":
            first_vol.setdefault(s, {})[d] = b.volume[0]
    rvol = {}
    for s, m in first_vol.items():
        ser = pd.Series(m).sort_index()
        base = ser.rolling(14, min_periods=10).mean().shift(1)
        rvol[s] = (ser / base).to_dict()

    cands = {k: {} for k in STRATEGIES}
    for d in days:
        ts = pd.Timestamp(d)
        if ts not in closes.index:
            continue
        rows = []
        for s in symbols:
            b = bars_by.get((s, d))
            rv = rvol.get(s, {}).get(d)
            if b is None or len(b) < 3 or rv is None or not np.isfinite(rv):
                continue
            pc, a, av = prev_c.at[ts, s], atr.at[ts, s], avg_vol.at[ts, s]
            if not (np.isfinite(pc) and np.isfinite(a) and np.isfinite(av)) or pc < config.MIN_PRICE:
                continue
            rows.append((s, b, rv, pc, a, av))

        sip = sorted([r for r in rows if r[2] >= S["MIN_RVOL_SIP"] and r[5] >= 1e6 and r[4] >= 0.5],
                     key=lambda r: -r[2])[:S["TOP_SIP"]]
        for rank, (s, b, rv, pc, a, av) in enumerate(sip, start=1):
            if b.hm[0] == "09:30" and b.close[0] != b.open[0]:
                up = b.close[0] > b.open[0]
                cands["sip_live"].setdefault(d, []).append({
                    "rank": rank, "symbol": s, "side": "long" if up else "short",
                    "level": float(b.high[0] if up else b.low[0]), "dist": S["SIP_STOP_ATR"] * float(a)})
            t = sip_trade(b, a)
            if t:
                cands["sip"].setdefault(d, []).append((b.hm[t["k"]], rank, s, t))
            t = vwap_trade(b, a)
            if t:
                cands["vwap"].setdefault(d, []).append((b.hm[t["k"]], rank, s, t))

        gaps = []
        for s, b, rv, pc, a, av in rows:
            g = (b.open[0] / pc - 1) * 100
            if abs(g) >= S["MIN_GAP_PCT"] and rv >= S["MIN_RVOL_GAP"]:
                gaps.append((abs(g), s, b, g))
        for rank, (_, s, b, g) in enumerate(sorted(gaps, key=lambda x: -x[0])[:S["TOP_GAP"]], start=1):
            t = gap_trade(b, g > 0)
            if t:
                cands["gap"].setdefault(d, []).append((b.hm[t["k"]], rank, s, t))
    return cands


class _View:
    """The bars of one day up to (not including) a cut-off index, for exit_from()."""

    def __init__(self, b, n):
        self.hm, self.open, self.high, self.low, self.close = b.hm[:n], b.open[:n], b.high[:n], b.low[:n], b.close[:n]

    def __len__(self):
        return len(self.hm)


def live_fill(b, side, level, p):
    """A stop entry order resting from bar p: (bar index, raw fill price), or None if it never fills before LIVE_ENTRY_UNTIL."""
    for j in range(p, len(b)):
        if b.hm[j] >= LIVE_ENTRY_UNTIL:
            return None
        if side == "long":
            if b.open[j] >= level:
                return j, float(b.open[j])        # already through the level: fills at the open
            if b.high[j] >= level:
                return j, float(level)
        else:
            if b.open[j] <= level:
                return j, float(b.open[j])
            if b.low[j] <= level:
                return j, float(level)
    return None


def live_exit(b, k, side, stop, entry_px):
    """The stop-loss, or the 3:55 PM market close-out (at the open of the 3:55 bar)."""
    slip = S["SLIPPAGE_PCT"] / 100
    cut = next((i for i in range(len(b)) if b.hm[i] >= LIVE_FLATTEN_AT), len(b))
    if k >= cut:
        return None
    hm_, px, why = exit_from(_View(b, cut), k, side, stop, None, entry_px)
    if why == "close" and cut < len(b):
        px = b.open[cut] * ((1 - slip) if side == "long" else (1 + slip))
        hm_ = b.hm[cut]
    return hm_, px, why


def simulate_live(days, picks_by_day, bars_by, spy, withdraw_pct):
    """Mirror sip_bot.py: entry orders go in rank order while open positions plus resting
    entries stay within MAX_INVESTED_PCT; room only frees up when a filled trade exits
    (the bot sees it on its next pass, i.e. from the next 5-minute bar); unfilled entries
    keep their room until LIVE_ENTRY_UNTIL; everything is closed at LIVE_FLATTEN_AT."""
    slip = S["SLIPPAGE_PCT"] / 100
    cash, withdrawn, month_pnl = START_CASH, 0.0, 0.0
    trades, equity_rows, months = [], [], []
    for di, d in enumerate(days):
        equity, day_pnl = cash, 0.0
        cap = equity * S["MAX_INVESTED_PCT"] / 100
        picks = sorted(picks_by_day.get(d, []), key=lambda p: p["rank"])
        used, nxt, now, placed = 0.0, 0, "09:35", 0
        releases = []   # (time the room frees up, notional)
        while True:
            for r in [r for r in releases if r[0] <= now]:
                used -= r[1]
                releases.remove(r)
            while nxt < len(picks) and now < LIVE_ENTRY_UNTIL:
                p = picks[nxt]
                b = bars_by.get((p["symbol"], d))
                if b is None or p["dist"] <= 0:
                    nxt += 1
                    continue
                qty = math.floor(min(equity * S["RISK_PCT"] / 100 / p["dist"],
                                     equity * S["MAX_POS_PCT"] / 100 / p["level"]))
                notional = qty * p["level"]
                if qty < 1:
                    nxt += 1
                    continue
                if used + notional > cap:
                    break
                used += notional
                nxt += 1
                placed += 1
                start = next((i for i in range(len(b)) if b.hm[i] >= now), None)
                f = live_fill(b, p["side"], p["level"], start) if start is not None else None
                if not f:
                    continue  # never filled: keeps its room until the entry cut-off
                k, raw = f
                long = p["side"] == "long"
                entry = raw * ((1 + slip) if long else (1 - slip))
                stop = raw - p["dist"] if long else raw + p["dist"]
                ex = live_exit(b, k, p["side"], stop, raw)
                if not ex:
                    continue
                xhm, xpx, why = ex
                pnl = (xpx - entry) * qty if long else (entry - xpx) * qty
                trades.append({"day": d, "symbol": p["symbol"], "side": p["side"], "entry_hm": str(b.hm[k]),
                               "exit_hm": str(xhm), "shares": qty, "entry": round(entry, 2), "stop": round(stop, 2),
                               "target": None, "exit": round(float(xpx), 2), "exit_reason": why,
                               "pnl": round(float(pnl), 2), "r_multiple": round(float(pnl / (p["dist"] * qty)), 2),
                               "rank": p["rank"]})
                day_pnl += pnl
                xi = list(b.hm).index(xhm)   # the bot sees the exit on its next pass
                releases.append((str(b.hm[xi + 1]) if xi + 1 < len(b) else "16:00", notional))
            future = sorted(r[0] for r in releases if r[0] > now)
            if nxt >= len(picks) or not future or future[0] >= LIVE_ENTRY_UNTIL:
                break
            now = future[0]
        cash += day_pnl
        month_pnl += day_pnl
        if di == len(days) - 1 or days[di + 1][:7] != d[:7]:
            take = max(0.0, month_pnl) * withdraw_pct / 100
            cash -= take
            withdrawn += take
            months.append({"month": d[:7], "realized_pl": round(month_pnl, 2), "withdrawn": round(take, 2),
                           "withdrawn_total": round(withdrawn, 2), "account_value": round(cash, 2)})
            month_pnl = 0.0
        equity_rows.append({"date": d, "equity": round(cash, 2), "trades": sum(t["day"] == d for t in trades),
                            "orders_placed": placed, "withdrawn_total": round(withdrawn, 2),
                            "spy_equity": round(spy[d], 2)})
    return trades, equity_rows, months, withdrawn


def simulate(days, cands_by_day, bars_by, max_trades, spy, withdraw_pct):
    slip = S["SLIPPAGE_PCT"] / 100
    cash, withdrawn, month_pnl = START_CASH, 0.0, 0.0
    trades, equity_rows, months = [], [], []
    for di, d in enumerate(days):
        equity, day_pnl, taken, open_t = cash, 0.0, 0, []
        for ehm, rank, s, t in sorted(cands_by_day.get(d, []), key=lambda c: (c[0], c[1])):
            if taken >= max_trades:
                break
            open_t = [x for x in open_t if x["exit_hm"] > ehm]
            long = t["side"] == "long"
            entry = t["entry_raw"] * ((1 + slip) if long else (1 - slip))
            risk = (entry - t["stop"]) if long else (t["stop"] - entry)
            if risk <= 0:
                continue
            qty = math.floor(min(equity * S["RISK_PCT"] / 100 / risk, equity * S["MAX_POS_PCT"] / 100 / entry))
            if qty < 1:
                continue
            if sum(x["entry"] * x["shares"] for x in open_t) + qty * entry > equity * S["MAX_INVESTED_PCT"] / 100:
                continue
            b = bars_by[(s, d)]
            xhm, xpx, why = exit_from(b, t["k"], t["side"], t["stop"], t["target"], t["entry_raw"])
            pnl = (xpx - entry) * qty if long else (entry - xpx) * qty
            rec = {"day": d, "symbol": s, "side": t["side"], "entry_hm": ehm, "exit_hm": str(xhm), "shares": qty,
                   "entry": round(entry, 2), "stop": round(t["stop"], 2),
                   "target": None if t["target"] is None else round(t["target"], 2),
                   "exit": round(float(xpx), 2), "exit_reason": why, "pnl": round(float(pnl), 2),
                   "r_multiple": round(float(pnl / (risk * qty)), 2)}
            trades.append(rec)
            open_t.append(rec)
            taken += 1
            day_pnl += pnl
        cash += day_pnl
        month_pnl += day_pnl
        if di == len(days) - 1 or days[di + 1][:7] != d[:7]:
            take = max(0.0, month_pnl) * withdraw_pct / 100
            cash -= take
            withdrawn += take
            months.append({"month": d[:7], "realized_pl": round(month_pnl, 2), "withdrawn": round(take, 2),
                           "withdrawn_total": round(withdrawn, 2), "account_value": round(cash, 2)})
            month_pnl = 0.0
        equity_rows.append({"date": d, "equity": round(cash, 2), "trades": taken,
                            "withdrawn_total": round(withdrawn, 2), "spy_equity": round(spy[d], 2)})
    return trades, equity_rows, months, withdrawn


def summarize(name, trades, eq_rows, months, withdrawn, withdraw_pct, max_trades):
    acct = pd.Series([r["equity"] for r in eq_rows])
    spy = pd.Series([r["spy_equity"] for r in eq_rows])
    dd = lambda s: round(float((s / s.cummax() - 1).min() * 100), 2)  # noqa: E731
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    gw, gl = sum(t["pnl"] for t in wins), -sum(t["pnl"] for t in losses)
    end = eq_rows[-1]["equity"]
    settings = {
        "sip": f"top {S['TOP_SIP']} by opening relative volume, stop {S['SIP_STOP_ATR']:g}x ATR, exit at close",
        "vwap": f"stocks in play, pullback to rising/falling VWAP 10:00-14:00, target {S['TARGET_R']:g}R",
        "sip_live": (f"the live bot rules: top {S['TOP_SIP']} by opening relative volume, entry orders in rank order "
                     f"while positions plus resting orders stay within the limit, entries until {LIVE_ENTRY_UNTIL}, "
                     f"stop {S['SIP_STOP_ATR']:g}x ATR, closed at {LIVE_FLATTEN_AT}"),
        "gap": f"gaps of {S['MIN_GAP_PCT']:g}%+ with {S['MIN_RVOL_GAP']:g}x opening volume, top {S['TOP_GAP']}, "
               f"break of first 5-min candle before 10:30, target {S['TARGET_R']:g}R",
    }[name] + (f"; risk {S['RISK_PCT']:g}% per trade, max {S['MAX_POS_PCT']:g}% per position, "
               f"max {max_trades} trades/day, max {S['MAX_INVESTED_PCT']:g}% invested, {S['SLIPPAGE_PCT']:g}% slippage")
    return {
        "start": eq_rows[0]["date"], "end": eq_rows[-1]["date"], "trading_days": len(eq_rows),
        "withdraw_pct": withdraw_pct, "start_value": START_CASH, "end_value": round(end, 2),
        "total_withdrawn": round(withdrawn, 2), "end_value_plus_withdrawn": round(end + withdrawn, 2),
        "total_return_incl_withdrawn_pct": round((end + withdrawn) / START_CASH * 100 - 100, 2),
        "max_drawdown_pct": dd(acct),
        "spy_end_value": round(float(spy.iloc[-1]), 2),
        "spy_total_return_pct": round(float(spy.iloc[-1]) / START_CASH * 100 - 100, 2),
        "spy_max_drawdown_pct": dd(spy),
        "trades": len(trades), "wins": len(wins), "losses": len(losses),
        "win_rate_pct": round(len(wins) / len(trades) * 100, 1) if trades else None,
        "avg_win": round(gw / len(wins), 2) if wins else None,
        "avg_loss": round(-gl / len(losses), 2) if losses else None,
        "profit_factor": round(gw / gl, 2) if gl else None,
        "avg_r": round(sum(t["r_multiple"] for t in trades) / len(trades), 3) if trades else None,
        "net_pl": round(sum(t["pnl"] for t in trades), 2),
        "months_positive": int(sum(float(m["realized_pl"]) > 0 for m in months)), "months": len(months),
        "settings": settings,
        "run_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Backtest professional day-trading methods")
    ap.add_argument("--years", type=int, default=1)
    a = ap.parse_args()

    symbols = orb.sp500_symbols()
    print(f"S&P 500 list: {len(symbols)} stocks")
    daily = orb.daily_data(symbols + ["SPY"], a.years)
    closes, opens = daily["Close"], daily["Open"]
    today = pd.Timestamp.now(tz=orb.NY).strftime("%Y-%m-%d")
    idx = [d for d in closes.index if d.strftime("%Y-%m-%d") < today]
    window = idx[-252 * a.years:]
    days = [d.strftime("%Y-%m-%d") for d in window]
    syms = [s for s in symbols if s in closes.columns]

    # Minute bars start ~1 month early so the 14-day opening-volume average is ready on day one
    start = (pd.Timestamp(days[0], tz=orb.NY) - pd.Timedelta(days=30)).to_pydatetime()
    end = (pd.Timestamp(days[-1], tz=orb.NY) + pd.Timedelta(hours=23)).to_pydatetime()
    bars_by = orb.day_bars(orb.minute_bars(syms, start, end))
    print(f"Test window {days[0]} to {days[-1]}: {len(days)} days")

    cands = build_candidates(days, bars_by, daily, syms)
    spy_shares = START_CASH / float(opens["SPY"].loc[window[0]])
    spy = {d.strftime("%Y-%m-%d"): spy_shares * float(closes["SPY"].loc[d]) for d in window}

    summ, eq_all, months_all, trades_all = [], [], [], []
    order = 0
    for name, (label, max_trades) in STRATEGIES.items():
        for wd in (0.0, 70.0):
            order += 1
            variant = name if wd == 0 else f"{name}_withdraw70"
            if name == "sip_live":
                trades, eq, months, withdrawn = simulate_live(days, cands[name], bars_by, spy, wd)
            else:
                trades, eq, months, withdrawn = simulate(days, cands[name], bars_by, max_trades, spy, wd)
            s = summarize(name, trades, eq, months, withdrawn, wd, max_trades)
            s.update(variant=variant, strategy=name, order=order,
                     label=label + (", 70% of profits withdrawn" if wd else ""))
            summ.append(s)
            eq_all += [dict(r, variant=variant) for r in eq]
            months_all += [dict(m, variant=variant) for m in months]
            if not wd:
                trades_all += [dict(t, variant=variant, id=f"{variant}-{i}")
                               for i, t in enumerate(sorted(trades, key=lambda t: (t["day"], t["entry_hm"]), reverse=True), 1)]
            print(f"\n== {s['label']} ==")
            print(to_json({k: s[k] for k in ("end_value_plus_withdrawn", "total_return_incl_withdrawn_pct",
                                                 "total_withdrawn", "max_drawdown_pct", "trades", "win_rate_pct",
                                                 "profit_factor", "avg_r", "months_positive", "months")}))

    OUT.mkdir(exist_ok=True)
    meta = [{"label": "backtest_daytrade.py", "value": Path(__file__).read_text(encoding="utf-8"), "type": "code", "lang": "python"},
            {"label": "backtest_orb.py (data download)", "value": (BASE / "backtest_orb.py").read_text(encoding="utf-8"), "type": "code", "lang": "python"},
            {"label": "Input", "value": "Alpaca 5-minute bars with volume and VWAP (split-adjusted); Yahoo Finance daily prices for S&P 500 stocks and SPY"}]
    note = [{"label": "Made by", "value": "backtest_daytrade.py (code attached to the summary source)"}]
    for fname, rows, m in (("dt_summary.json", summ, meta), ("dt_equity.json", eq_all, note),
                           ("dt_months.json", months_all, note), ("dt_trades.json", trades_all, note)):
        (OUT / fname).write_text(to_json({"rows": rows, "meta": m}, separators=(",", ":")))


if __name__ == "__main__":
    main()
