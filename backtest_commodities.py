"""
Backtest gold and oil (through the GLD and USO ETFs, which the Alpaca account can trade) with the
day trader's know-how plus two methods suited to commodities, and pick each market's best version
honestly: the pick is made on the FIRST half of the year only, then judged on the SECOND half,
which it never saw.

Methods tried on each market (32 versions):
  orb    The live day trader's rules on one ETF: first 5-minute candle breakout in its direction,
         entries until 3:30 PM, stop k x ATR, optional stop-to-breakeven at +1R, closed at 3:55 PM,
         optionally only on days with unusual opening volume.      k x breakeven x volume = 18
  gap    Fade a big opening gap back toward yesterday's close (target), stop s x ATR, closed at 3:55.
                                                                     gap size x stop      =  6
  trend  Daily trend following, positions held overnight: channel breakouts (Donchian) or moving
         average crosses, long only or long and short.             4 rules x 2           =  8

Sizing follows the live bot: 1% of the account at risk per day trade, at most 20% of the account in
one position (trend trades simply hold 20%); 0.05% slippage on every fill. Each market starts with
$100,000 on its own.

Data: Alpaca 5-minute bars (shared cache with the other backtests), Yahoo Finance daily prices.
Output: backtest_commodities_results/cmd_{summary,equity,months,trades,grid}.json
Usage: python backtest_commodities.py [--years 1]
Reads historical prices only; never places orders.
"""
import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import backtest_daytrade as dt
import backtest_orb as orb

BASE = Path(__file__).resolve().parent
OUT = BASE / "backtest_commodities_results"
MARKETS = {"GLD": "Gold (GLD)", "USO": "Oil (USO)", "GDX": "Gold miners (GDX)"}
START_CASH = 100_000.0
RISK_PCT, MAX_POS_PCT, SLIP_PCT = 1.0, 20.0, 0.05
FLATTEN_AT, ENTRY_UNTIL = "15:55", "15:30"


# ---------------------------------------------------------------------------
# Versions
# ---------------------------------------------------------------------------
def versions() -> list[dict]:
    out = []
    for k in (0.05, 0.10, 0.20):
        for be in (None, 1.0):
            for rv in (0.0, 1.5, 2.0):
                out.append({"method": "orb", "stop_atr": k, "be_r": be, "min_rvol": rv,
                            "name": f"orb_k{k:g}_{'be' if be else 'nobe'}_rv{rv:g}",
                            "label": (f"5-min breakout, stop {k:g}x ATR" + (", breakeven at +1R" if be else "")
                                      + (f", opening volume {rv:g}x+" if rv else ", every day"))})
    for g in (0.3, 0.5, 0.75):
        for s in (0.3, 0.5):
            out.append({"method": "gap", "gap_atr": g, "stop_atr": s, "name": f"gap_g{g:g}_s{s:g}",
                        "label": f"Gap fade: gaps of {g:g}x ATR+, stop {s:g}x ATR, target yesterday's close"})
    for rule in (("donch", 20, 10), ("donch", 55, 20), ("ma", 20, 50), ("ma", 50, 200)):
        for ls in (False, True):
            kind, a, b = rule
            what = f"{a}-day high/{b}-day low channel" if kind == "donch" else f"{a}/{b}-day moving average cross"
            out.append({"method": "trend", "rule": kind, "a": a, "b": b, "short": ls,
                        "name": f"trend_{kind}{a}_{b}_{'ls' if ls else 'long'}",
                        "label": f"Trend: {what}, {'long and short' if ls else 'long only'}, held overnight"})
    return out


# ---------------------------------------------------------------------------
# Day trades
# ---------------------------------------------------------------------------
def day_exit(b, k, side, stop, target, entry_raw, be_r):
    """Stop, target or the 3:55 PM close-out, using the day-trade backtest's bar-path rules."""
    saved = dict(dt.S)
    dt.S.update(SLIPPAGE_PCT=SLIP_PCT, BE_R=be_r)
    try:
        slip = SLIP_PCT / 100
        cut = next((i for i in range(len(b)) if b.hm[i] >= FLATTEN_AT), len(b))
        if k >= cut:
            return None
        hm_, px, why = dt.exit_from(dt._View(b, cut), k, side, stop, target, entry_raw)
        if why == "close" and cut < len(b):
            px, hm_ = b.open[cut] * ((1 - slip) if side == "long" else (1 + slip)), b.hm[cut]
        return str(hm_), float(px), why
    finally:
        dt.S.clear()
        dt.S.update(saved)


def day_signal(v, b, atr, prev_close, rvol):
    if b is None or len(b) < 3 or b.hm[0] != "09:30" or not np.isfinite(atr) or atr <= 0:
        return None
    if v["method"] == "orb":
        if rvol is None or not np.isfinite(rvol) or rvol < v["min_rvol"]:
            return None
        nb = v.get("or_bars", 1)
        if len(b) <= nb:
            return None
        o, c = b.open[0], b.close[nb - 1]
        if c == o:
            return None
        side = "long" if c > o else "short"
        level = b.high[:nb].max() if side == "long" else b.low[:nb].min()
        hit = (b.high[nb:] >= level) if side == "long" else (b.low[nb:] <= level)
        hit &= b.hm[nb:] < ENTRY_UNTIL
        idx = np.flatnonzero(hit)
        if not len(idx):
            return None
        k = int(idx[0]) + nb
        entry = max(level, b.open[k]) if side == "long" else min(level, b.open[k])
        d = v["stop_atr"] * atr
        return {"side": side, "k": k, "entry_raw": float(entry),
                "stop": float(entry - d if side == "long" else entry + d), "target": None}
    if v["method"] == "gap":
        if not np.isfinite(prev_close):
            return None
        g = (b.open[0] - prev_close) / atr
        if abs(g) < v["gap_atr"]:
            return None
        side = "short" if g > 0 else "long"
        entry = float(b.open[0])
        d = v["stop_atr"] * atr
        return {"side": side, "k": 0, "entry_raw": entry,
                "stop": float(entry - d if side == "long" else entry + d), "target": float(prev_close)}
    return None


def run_day_trades(v, sym, days, ctx):
    slip = SLIP_PCT / 100
    eq, trades, rows = START_CASH, [], []
    for d in days:
        b = ctx["bars"].get((sym, d))
        t = day_signal(v, b, ctx["atr"][sym].get(d, np.nan), ctx["prev_close"][sym].get(d, np.nan),
                       ctx["rvol"][sym].get(d))
        n = 0
        if t:
            long = t["side"] == "long"
            entry = t["entry_raw"] * ((1 + slip) if long else (1 - slip))
            risk = (entry - t["stop"]) if long else (t["stop"] - entry)
            qty = math.floor(min(eq * RISK_PCT / 100 / risk, eq * MAX_POS_PCT / 100 / entry)) if risk > 0 else 0
            ex = day_exit(b, t["k"], t["side"], t["stop"], t["target"], t["entry_raw"], v.get("be_r")) if qty >= 1 else None
            if ex:
                xhm, xpx, why = ex
                pnl = (xpx - entry) * qty if long else (entry - xpx) * qty
                eq += pnl
                n = 1
                trades.append({"day": d, "symbol": sym, "side": t["side"], "entry_hm": str(b.hm[t["k"]]), "exit_hm": xhm,
                               "shares": qty, "entry": round(entry, 2), "stop": round(t["stop"], 2),
                               "target": None if t["target"] is None else round(t["target"], 2),
                               "exit": round(xpx, 2), "exit_reason": why, "pnl": round(pnl, 2),
                               "r_multiple": round(pnl / (risk * qty), 2)})
        rows.append({"date": d, "equity": round(eq, 2), "trades": n})
    return trades, rows


# ---------------------------------------------------------------------------
# Trend trades (daily, held overnight; signal at the close, trade at the next open)
# ---------------------------------------------------------------------------
def trend_targets(v, hi, lo, cl):
    """Desired position (+1 long, -1 short, 0 flat) after each day's close."""
    pos, out = 0, []
    if v["rule"] == "donch":
        up, dn = hi.shift(1).rolling(v["a"]).max(), lo.shift(1).rolling(v["a"]).min()
        xl, xs = lo.shift(1).rolling(v["b"]).min(), hi.shift(1).rolling(v["b"]).max()
        for c, u, dd, a, s in zip(cl, up, dn, xl, xs):
            if pos == 1 and c < a:
                pos = 0
            elif pos == -1 and c > s:
                pos = 0
            if pos == 0 and c > u:
                pos = 1
            elif pos == 0 and v["short"] and c < dd:
                pos = -1
            out.append(pos)
    else:
        f, s = cl.rolling(v["a"]).mean(), cl.rolling(v["b"]).mean()
        for x, y in zip(f, s):
            out.append(0 if not (np.isfinite(x) and np.isfinite(y)) else (1 if x > y else (-1 if v["short"] else 0)))
    return pd.Series(out, index=cl.index)


def run_trend(v, sym, days, ctx):
    slip = SLIP_PCT / 100
    op, hi, lo, cl = (ctx["daily"][f][sym].dropna() for f in ("Open", "High", "Low", "Close"))
    want = trend_targets(v, hi, lo, cl)
    idx = {d.strftime("%Y-%m-%d"): i for i, d in enumerate(cl.index)}
    eq, pos, qty, entry, entry_day, trades, rows = START_CASH, 0, 0, 0.0, None, [], []
    for d in days:
        i = idx.get(d)
        if i is None or i == 0:
            rows.append({"date": d, "equity": round(eq, 2), "trades": 0})
            continue
        o, c, pc = float(op.iloc[i]), float(cl.iloc[i]), float(cl.iloc[i - 1])
        target = int(want.iloc[i - 1])
        n = 0
        if pos and target != pos:           # close at today's open
            px = o * ((1 - slip) if pos == 1 else (1 + slip))
            eq += (px - pc) * qty * pos
            pnl = (px - entry) * qty * pos
            trades.append({"day": entry_day, "symbol": sym, "side": "long" if pos == 1 else "short", "entry_hm": "09:30",
                           "exit_hm": d, "shares": qty, "entry": round(entry, 2), "stop": None, "target": None,
                           "exit": round(px, 2), "exit_reason": "signal", "pnl": round(pnl, 2),
                           "r_multiple": round(pnl / (entry * qty * 0.02), 2)})   # R = a 2% move of the position
            pos, qty, n = 0, 0, n + 1
            mark = None
        else:
            mark = pc if pos else None
        if not pos and target:              # open at today's open
            entry = o * ((1 + slip) if target == 1 else (1 - slip))
            qty = math.floor(eq * MAX_POS_PCT / 100 / entry)
            if qty >= 1:
                pos, entry_day, n, mark = target, d, n + 1, entry
        if pos:
            eq += (c - mark) * qty * pos
        rows.append({"date": d, "equity": round(eq, 2), "trades": n})
    if pos:                                  # still open at the end: count it at the last close
        c = float(cl.loc[pd.Timestamp(days[-1])]) if pd.Timestamp(days[-1]) in cl.index else float(cl.iloc[-1])
        pnl = (c - entry) * qty * pos
        trades.append({"day": entry_day, "symbol": sym, "side": "long" if pos == 1 else "short", "entry_hm": "09:30",
                       "exit_hm": "open", "shares": qty, "entry": round(entry, 2), "stop": None, "target": None,
                       "exit": round(c, 2), "exit_reason": "still open", "pnl": round(pnl, 2),
                       "r_multiple": round(pnl / (entry * qty * 0.02), 2)})
    return trades, rows


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
def months_of(rows):
    out, prev = [], START_CASH
    for i, r in enumerate(rows):
        if i == len(rows) - 1 or rows[i + 1]["date"][:7] != r["date"][:7]:
            out.append({"month": r["date"][:7], "realized_pl": round(r["equity"] - prev, 2), "withdrawn": 0.0,
                        "withdrawn_total": 0.0, "account_value": r["equity"]})
            prev = r["equity"]
    return out


def summary_row(trades, rows, spy, settings):
    """The same fields as the day-trade backtest's summary rows."""
    acct = pd.Series([r["equity"] for r in rows])
    sp = pd.Series([spy[r["date"]] for r in rows])
    dd = lambda s: round(float((s / s.cummax() - 1).min() * 100), 2)  # noqa: E731
    closed = [t for t in trades if t["exit_reason"] != "still open"] or trades
    wins = [t for t in closed if t["pnl"] > 0]
    losses = [t for t in closed if t["pnl"] <= 0]
    gw, gl = sum(t["pnl"] for t in wins), -sum(t["pnl"] for t in losses)
    mo = months_of(rows)
    end = rows[-1]["equity"]
    return {
        "start": rows[0]["date"], "end": rows[-1]["date"], "trading_days": len(rows), "withdraw_pct": 0.0,
        "start_value": START_CASH, "end_value": round(end, 2), "total_withdrawn": 0.0,
        "end_value_plus_withdrawn": round(end, 2), "total_return_incl_withdrawn_pct": round(end / START_CASH * 100 - 100, 2),
        "max_drawdown_pct": dd(acct), "spy_end_value": round(float(sp.iloc[-1]), 2),
        "spy_total_return_pct": round(float(sp.iloc[-1]) / START_CASH * 100 - 100, 2), "spy_max_drawdown_pct": dd(sp),
        "trades": len(trades), "wins": len(wins), "losses": len(losses),
        "win_rate_pct": round(len(wins) / len(closed) * 100, 1) if closed else None,
        "avg_win": round(gw / len(wins), 2) if wins else None, "avg_loss": round(-gl / len(losses), 2) if losses else None,
        "profit_factor": round(gw / gl, 2) if gl else None,
        "avg_r": round(sum(t["r_multiple"] for t in trades) / len(trades), 3) if trades else None,
        "net_pl": round(sum(t["pnl"] for t in trades), 2),
        "months_positive": int(sum(m["realized_pl"] > 0 for m in mo)), "months": len(mo),
        "settings": settings, "run_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def halves(rows):
    """Return in the first half (where the pick is made) and the second half (the honest test)."""
    mid = len(rows) // 2
    m = rows[mid - 1]["equity"]
    return round((m / START_CASH - 1) * 100, 2), round((rows[-1]["equity"] / m - 1) * 100, 2), rows[mid]["date"]


def pick(results, method_min_trades=None):
    """Best first-half return among versions with enough first-half trades."""
    method_min_trades = method_min_trades or MIN_FIRST_HALF_TRADES
    ok = [r for r in results if r["first_half_trades"] >= method_min_trades[r["method"]]]
    return max(ok, key=lambda r: r["first_half_pct"]) if ok else None


def combine(parts, spy, settings):
    """Several accounts' daily results added together on one $100,000 account (each part's daily
    % change is applied to the shared account). parts: list of daily row lists; a day missing from a part counts as no change."""
    by = [{r["date"]: r for r in p} for p in parts]
    dates = [r["date"] for r in parts[0]]
    eq, rows = START_CASH, []
    prev = [START_CASH] * len(parts)
    for d in dates:
        ch = 0.0
        n = 0
        for j, p in enumerate(by):
            r = p.get(d)
            if r is None:
                continue
            ch += r["equity"] / prev[j] - 1
            prev[j] = r["equity"]
            n += r.get("trades", 0)
        eq *= 1 + ch
        rows.append({"date": d, "equity": round(eq, 2), "trades": n})
    return rows


# ---------------------------------------------------------------------------
# More methods (Nov 18, 2026): the gold methods traders teach, plus ones with published evidence
# ---------------------------------------------------------------------------
# FOMC statement days (federalreserve.gov meeting calendar; the statement comes at 2:00 PM)
FOMC_DAYS = {"2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30", "2025-09-17", "2025-10-29",
             "2025-12-10", "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17", "2026-07-29", "2026-09-16",
             "2026-10-28", "2026-12-09"}


def more_versions() -> list[dict]:
    out = []
    # --- 1. Trend following and pullbacks (daily): trade with the trend, enter on a dip to a moving average
    for f, s in ((50, 200), (20, 100)):
        for pe in (20, 50):
            for r in (2.0, 3.0):
                out.append({"method": "pullback", "fast": f, "slow": s, "pe": pe, "take_r": r,
                            "name": f"pullback_{f}_{s}_ema{pe}_{r:g}r",
                            "label": f"Trend pullback: {f}/{s}-day EMA trend, buy dips to the {pe}-day EMA (short the mirror), "
                                     f"stop below the dip, target {r:g}R, at most 20 days"})
    for ls in (False, True):
        out.append({"method": "fib", "short": ls, "name": f"fib_{'ls' if ls else 'long'}",
                    "label": "Fibonacci pullback: in a 50/200-day EMA trend, buy a 50-61.8% retracement of the last 20 days' "
                             f"swing, stop at 78.6%, target the swing high{' (short the mirror)' if ls else ', long only'}"})
    # --- 2. Breakouts and market structure (intraday)
    for nb in (3, 6):
        for be in (None, 1.0):
            out.append({"method": "orb", "or_bars": nb, "stop_atr": 0.10, "be_r": be, "min_rvol": 0.0,
                        "name": f"orb{5 * nb}_{'be' if be else 'nobe'}",
                        "label": f"{5 * nb}-min opening range breakout, stop 0.1x ATR" + (", breakeven at +1R" if be else "")})
    for kind in ("strict", "loose"):
        out.append({"method": "sweep", "kind": kind, "name": f"sweep_{kind}",
                    "label": "Liquidity sweep of yesterday's high/low, then structure shift and fair value gap entry "
                             f"({'strict' if kind == 'strict' else 'looser'} rules), 2R target"})
    # --- 3. Range trading (daily oscillators, only when there is no trend)
    for osc, n, lo, hi in (("rsi", 2, 10, 90), ("rsi", 14, 30, 70), ("stoch", 14, 20, 80)):
        for adx in (True, False):
            out.append({"method": "range", "osc": osc, "n": n, "lo": lo, "hi": hi, "adx": adx,
                        "name": f"range_{osc}{n}_{'adx' if adx else 'any'}",
                        "label": f"Range trading: buy {osc.upper()}({n}) below {lo}, short above {hi}"
                                 + (" when ADX < 20 (no trend)" if adx else " on any day")
                                 + ", exit at 50, stop 2x ATR, at most 10 days"})
    # --- 4. Macro and news
    out.append({"method": "fomc_drift", "name": "fomc_drift", "fill": "close",
                "label": "Fed days: hold from the close before each FOMC statement to the close on statement day"})
    out.append({"method": "fomc_break", "be_r": None, "name": "fomc_break",
                "label": "Fed days: trade the breakout of the 2:00-2:30 PM range after the FOMC statement, closed by 3:55"})
    for short in (False, True):
        out.append({"method": "dollar", "short": short, "name": f"dollar_{'ls' if short else 'long'}",
                    "label": "Dollar filter: in a 20/50-day uptrend while the dollar (UUP) is below its 50-day average"
                             + (" (short when both reverse)" if short else ", long only")})
    for drop in (1.5, 2.5):
        out.append({"method": "haven", "drop": drop, "fill": "close", "name": f"haven_{drop:g}",
                    "label": f"Safe haven: buy at the close when the S&P 500 (SPY) falls {drop:g}%+ in a day, hold 5 days"})
    # --- 5. Methods with published evidence
    for flt in (False, True):
        for slip in (0.05, 0.01):
            out.append({"method": "overnight", "filter": flt, "slip": slip, "fill": "close",
                        "name": f"overnight_{'trend' if flt else 'all'}_s{slip:g}",
                        "label": "Overnight drift: buy at the close, sell at the next open"
                                 + (" (only above the 50-day average)" if flt else " every day")
                                 + f", {slip:g}% slippage" + (" (closing/opening auction orders)" if slip < 0.05 else "")})
    for months in (3, 6, 12):
        out.append({"method": "tsmom", "months": months, "fill": "close", "name": f"tsmom_{months}m",
                    "label": f"Time-series momentum: at each month end, hold if up over the last {months} months, else cash"})
    out.append({"method": "tsmom", "months": 0, "fill": "close", "name": "tsmom_10m_avg",
                "label": "Monthly trend filter: at each month end, hold if above the 10-month average, else cash"})
    out.append({"method": "preholiday", "fill": "close", "name": "preholiday",
                "label": "Pre-holiday: buy at the close two sessions before a market holiday, sell at the close the day before"})
    for n in (50, 100):
        out.append({"method": "rotation", "n": n, "name": f"rotation_{n}",
                    "label": f"Gold vs S&P 500: hold when the gold/SPY price ratio is above its {n}-day average, else cash"})
    return out


MIN_FIRST_HALF_TRADES = {"orb": 15, "gap": 8, "trend": 1, "pullback": 3, "fib": 2, "sweep": 5, "range": 3,
                         "fomc_drift": 2, "fomc_break": 2, "dollar": 1, "haven": 2, "overnight": 20, "tsmom": 1,
                         "preholiday": 2, "rotation": 1}


def indicators(daily, sym):
    o, h, l, c = (daily[f][sym].dropna() for f in ("Open", "High", "Low", "Close"))
    h, l, o = h.reindex(c.index), l.reindex(c.index), o.reindex(c.index)
    ema = lambda n: c.ewm(span=n, adjust=False).mean()  # noqa: E731
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()

    def rsi(n):
        d = c.diff()
        up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
        dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
        return 100 - 100 / (1 + up / dn.replace(0, np.nan))

    stoch = (c - l.rolling(14).min()) / (h.rolling(14).max() - l.rolling(14).min()) * 100
    upm, dnm = h.diff(), -l.diff()
    pdm = pd.Series(np.where((upm > dnm) & (upm > 0), upm, 0.0), index=c.index)
    ndm = pd.Series(np.where((dnm > upm) & (dnm > 0), dnm, 0.0), index=c.index)
    atr_w = tr.ewm(alpha=1 / 14, adjust=False).mean()
    pdi = 100 * pdm.ewm(alpha=1 / 14, adjust=False).mean() / atr_w
    ndi = 100 * ndm.ewm(alpha=1 / 14, adjust=False).mean() / atr_w
    adx = (100 * (pdi - ndi).abs() / (pdi + ndi)).ewm(alpha=1 / 14, adjust=False).mean()
    spy = daily["Close"]["SPY"].reindex(c.index)
    uup = daily["Close"]["UUP"].reindex(c.index).ffill() if "UUP" in daily["Close"] else pd.Series(np.nan, index=c.index)
    return {"o": o, "h": h, "l": l, "c": c, "atr": atr, "ema": {n: ema(n) for n in (20, 50, 100, 200)},
            "sma50": c.rolling(50).mean(), "rsi": {2: rsi(2), 14: rsi(14)}, "stoch": stoch, "adx": adx,
            "spy": spy, "spy_ret": spy.pct_change() * 100, "uup": uup, "uup50": uup.rolling(50).mean(),
            "ratio": c / spy}


def decide(v, i, I, st):
    """Called after day i's close. st holds the open position ("pos", "held" = days held).
    Returns None (no change) or an order {"pos": -1/0/1, "stop", "take" or "take_r", "exit_at_open"}."""
    c, h, l, atr = I["c"].iloc[i], I["h"].iloc[i], I["l"].iloc[i], I["atr"].iloc[i]
    pos, m = st["pos"], v["method"]
    if not np.isfinite(atr):
        return None
    if m == "pullback":
        f, s, e = I["ema"][v["fast"]].iloc[i], I["ema"][v["slow"]].iloc[i], I["ema"][v["pe"]].iloc[i]
        up, down = f > s, f < s
        if (pos == 1 and (not up or st["held"] >= 20)) or (pos == -1 and (not down or st["held"] >= 20)):
            return {"pos": 0}
        if pos == 0 and up and l <= e < c:
            return {"pos": 1, "stop": l - 0.5 * atr, "take_r": v["take_r"]}
        if pos == 0 and down and h >= e > c:
            return {"pos": -1, "stop": h + 0.5 * atr, "take_r": v["take_r"]}
        return None
    if m == "fib":
        if i < 20:
            return None
        up = I["ema"][50].iloc[i] > I["ema"][200].iloc[i]
        hi20, lo20 = I["h"].iloc[i - 20:i].max(), I["l"].iloc[i - 20:i].min()
        rng = hi20 - lo20
        if pos != 0:
            return {"pos": 0} if st["held"] >= 20 else None
        if up and l <= hi20 - 0.5 * rng and c >= hi20 - 0.618 * rng:
            return {"pos": 1, "stop": hi20 - 0.786 * rng, "take": hi20}
        if v["short"] and not up and h >= lo20 + 0.5 * rng and c <= lo20 + 0.618 * rng:
            return {"pos": -1, "stop": lo20 + 0.786 * rng, "take": lo20}
        return None
    if m == "range":
        x = I["rsi"][v["n"]].iloc[i] if v["osc"] == "rsi" else I["stoch"].iloc[i]
        if (pos == 1 and (x >= 50 or st["held"] >= 10)) or (pos == -1 and (x <= 50 or st["held"] >= 10)):
            return {"pos": 0}
        if pos == 0 and (not v["adx"] or I["adx"].iloc[i] < 20):
            if x < v["lo"]:
                return {"pos": 1, "stop": c - 2 * atr}
            if x > v["hi"]:
                return {"pos": -1, "stop": c + 2 * atr}
        return None
    if m == "fomc_drift":
        if pos:
            return {"pos": 0}
        nxt = I["c"].index[i + 1].strftime("%Y-%m-%d") if i + 1 < len(I["c"]) else None
        return {"pos": 1} if nxt in FOMC_DAYS else None
    if m == "dollar":
        gu = I["ema"][20].iloc[i] > I["ema"][50].iloc[i]
        du = I["uup"].iloc[i] > I["uup50"].iloc[i]
        want = 1 if gu and not du else (-1 if v["short"] and not gu and du else 0)
        return {"pos": want} if want != pos else None
    if m == "haven":
        if pos:
            return {"pos": 0} if st["held"] >= 5 else None
        return {"pos": 1} if I["spy_ret"].iloc[i] <= -v["drop"] else None
    if m == "overnight":
        ok = (not v["filter"]) or c > I["sma50"].iloc[i]
        return {"pos": 1, "exit_at_open": True} if ok else None
    if m == "tsmom":
        idx = I["c"].index
        if i + 1 < len(idx) and idx[i + 1].month == idx[i].month:
            return None                                   # only at month end
        if v["months"]:
            past = I["c"].loc[:idx[i] - pd.DateOffset(months=v["months"])]
            if past.empty:
                return None
            up = c > past.iloc[-1]
        else:
            mon = I["c"].loc[:idx[i]].resample("ME").last()
            if len(mon) < 10:
                return None
            up = c > mon.iloc[-10:].mean()
        want = 1 if up else 0
        return {"pos": want} if want != pos else None
    if m == "preholiday":
        idx = I["c"].index
        if pos:
            return {"pos": 0}
        if i + 2 < len(idx) and np.busday_count(idx[i + 1].date(), idx[i + 2].date()) > 1:
            return {"pos": 1}                             # a weekday market holiday follows day i+1
        return None
    if m == "rotation":
        r = I["ratio"]
        avg = r.rolling(v["n"]).mean().iloc[i]
        want = 1 if r.iloc[i] > avg else 0
        return {"pos": want} if np.isfinite(avg) and want != pos else None
    return None


def run_daily(v, sym, days, ctx):
    """Daily strategies. Orders go in at the next open, or in the same day's closing auction for
    "fill": "close" methods; stops and targets are checked on each day's high and low (stop first)."""
    slip = v.get("slip", SLIP_PCT) / 100
    I = ctx["ind"][sym]
    c_ser = I["c"]
    pos_of = {d.strftime("%Y-%m-%d"): i for i, d in enumerate(c_ser.index)}
    st = {"pos": 0, "held": 0, "qty": 0, "entry": 0.0, "entry_day": None, "entry_hm": None,
          "stop": None, "take": None, "exit_at_open": False}
    acct = {"cash": START_CASH}
    trades, rows, pending = [], [], None

    def size(px, stop):
        eq = acct["cash"]
        if stop is not None and abs(px - stop) > 0:
            return math.floor(min(eq * RISK_PCT / 100 / abs(px - stop), eq * MAX_POS_PCT / 100 / px))
        return math.floor(eq * MAX_POS_PCT / 100 / px)

    def close_pos(d, px, why):
        p = st["pos"]
        fill = px * ((1 - slip) if p == 1 else (1 + slip))
        pnl = (fill - st["entry"]) * st["qty"] * p
        acct["cash"] += pnl
        risk = abs(st["entry"] - st["stop"]) * st["qty"] if st["stop"] is not None else st["entry"] * st["qty"] * 0.02
        trades.append({"day": st["entry_day"], "symbol": sym, "side": "long" if p == 1 else "short",
                       "entry_hm": st["entry_hm"], "exit_hm": d, "shares": st["qty"], "entry": round(st["entry"], 2),
                       "stop": None if st["stop"] is None else round(st["stop"], 2),
                       "target": None if st["take"] is None else round(st["take"], 2),
                       "exit": round(fill, 2), "exit_reason": why, "pnl": round(pnl, 2),
                       "r_multiple": round(pnl / risk, 2) if risk else None})
        st.update(pos=0, qty=0, held=0, stop=None, take=None, exit_at_open=False)

    def open_pos(d, px, order, when):
        p = order["pos"]
        fill = px * ((1 + slip) if p == 1 else (1 - slip))
        stop = order.get("stop")
        if stop is not None and ((p == 1 and stop >= fill) or (p == -1 and stop <= fill)):
            return                                        # the price already went through the stop
        qty = size(fill, stop)
        if qty < 1:
            return
        take = order.get("take")
        if order.get("take_r") and stop is not None:
            take = fill + order["take_r"] * (fill - stop) if p == 1 else fill - order["take_r"] * (stop - fill)
        if take is not None and ((p == 1 and take <= fill) or (p == -1 and take >= fill)):
            return                                        # the target is already behind the entry
        st.update(pos=p, qty=qty, entry=fill, entry_day=d, entry_hm=when, stop=stop, take=take, held=0,
                  exit_at_open=bool(order.get("exit_at_open")))

    c = None
    for d in days:
        i = pos_of.get(d)
        if i is None:
            rows.append({"date": d, "equity": round(acct["cash"], 2), "trades": 0})
            continue
        o, h, l, c = (float(I[k].iloc[i]) for k in ("o", "h", "l", "c"))
        n0 = len(trades)
        # 1) the open: overnight exits, then orders decided at yesterday's close
        if st["pos"] and st["exit_at_open"]:
            close_pos(d, o, "open")
        if pending is not None:
            if st["pos"] and pending["pos"] != st["pos"]:
                close_pos(d, o, "signal")
            if pending["pos"] and not st["pos"]:
                open_pos(d, o, pending, "09:30")
            pending = None
        # 2) during the day: a gap through the stop fills at the open; else stop first, then target
        if st["pos"]:
            p, sp, tk = st["pos"], st["stop"], st["take"]
            fresh = st["entry_day"] == d and st["entry_hm"] == "16:00"
            if not fresh:
                if sp is not None and ((p == 1 and o <= sp) or (p == -1 and o >= sp)):
                    close_pos(d, o, "stop")
                elif sp is not None and ((p == 1 and l <= sp) or (p == -1 and h >= sp)):
                    close_pos(d, sp, "stop")
                elif tk is not None and ((p == 1 and h >= tk) or (p == -1 and l <= tk)):
                    close_pos(d, max(tk, o) if p == 1 else min(tk, o), "target")
        # 3) the close: decide; closing-auction methods trade now, the others at the next open
        if st["pos"]:
            st["held"] += 1
        order = decide(v, i, I, st)
        if order is not None:
            if v.get("fill") == "close":
                if st["pos"] and order["pos"] != st["pos"]:
                    close_pos(d, c, "signal")
                if order["pos"] and not st["pos"]:
                    open_pos(d, c, order, "16:00")
            else:
                pending = order
        eqv = acct["cash"] + (st["qty"] * (c - st["entry"]) * st["pos"] if st["pos"] else 0.0)
        rows.append({"date": d, "equity": round(eqv, 2), "trades": len(trades) - n0})
    if st["pos"] and c is not None:
        p = st["pos"]
        pnl = (c - st["entry"]) * st["qty"] * p
        trades.append({"day": st["entry_day"], "symbol": sym, "side": "long" if p == 1 else "short",
                       "entry_hm": st["entry_hm"], "exit_hm": "open", "shares": st["qty"], "entry": round(st["entry"], 2),
                       "stop": None if st["stop"] is None else round(st["stop"], 2),
                       "target": None if st["take"] is None else round(st["take"], 2), "exit": round(c, 2),
                       "exit_reason": "still open", "pnl": round(pnl, 2), "r_multiple": None})
    return trades, rows


def run_intraday_more(v, sym, days, ctx):
    """Liquidity sweeps (the TJR model on yesterday's high/low) and the FOMC 2:00-2:30 PM breakout."""
    import backtest_tjr as tjr

    slip = SLIP_PCT / 100
    eq, trades, rows = START_CASH, [], []
    for d in days:
        b = ctx["bars"].get((sym, d))
        t = None
        if b is not None and len(b) > 20 and v["method"] == "sweep":
            saved = dict(tjr.P)
            tjr.P.update(tjr.STRICT if v["kind"] == "strict" else dict(tjr.STRICT, RECLAIM_BARS=3, MSS_BARS=24, FVG_ATR=0.10))
            try:
                df = pd.DataFrame({"hm": b.hm, "open": b.open, "high": b.high, "low": b.low, "close": b.close})
                pools = {"low": [ctx["prev_low"][sym].get(d, np.nan)], "high": [ctx["prev_high"][sym].get(d, np.nan)]}
                pools = {k: [x for x in xs if np.isfinite(x)] for k, xs in pools.items()}
                for s_ in tjr.find_setups(df, pools):
                    r = tjr.fill_and_exit(df, s_)
                    if r:
                        ehm, fill, xhm, xpx, why = r
                        long = s_["side"] == "long"
                        t = {"side": s_["side"], "entry_hm": ehm, "entry_raw": float(fill), "stop": float(s_["stop"]),
                             "target": float(s_["target"]), "exit_hm": xhm,
                             "exit_px": float(xpx) * ((1 - slip) if long else (1 + slip)) if why != "stop" else float(xpx),
                             "why": why}
                        break
            finally:
                tjr.P.clear()
                tjr.P.update(saved)
        elif b is not None and len(b) > 20 and v["method"] == "fomc_break" and d in FOMC_DAYS:
            k0 = next((i for i in range(len(b)) if b.hm[i] >= "14:00"), None)
            k1 = next((i for i in range(len(b)) if b.hm[i] >= "14:30"), None)
            if k0 is not None and k1 is not None and k1 > k0:
                hi, lo = float(b.high[k0:k1].max()), float(b.low[k0:k1].min())
                for k in range(k1, len(b)):
                    if b.hm[k] >= ENTRY_UNTIL:
                        break
                    side = "long" if b.high[k] >= hi else ("short" if b.low[k] <= lo else None)
                    if side:
                        entry = max(hi, b.open[k]) if side == "long" else min(lo, b.open[k])
                        stop = lo if side == "long" else hi
                        ex = day_exit(b, k, side, stop, None, float(entry), v.get("be_r"))
                        if ex:
                            t = {"side": side, "entry_hm": str(b.hm[k]), "entry_raw": float(entry), "stop": stop,
                                 "target": None, "exit_hm": ex[0], "exit_px": ex[1], "why": ex[2]}
                        break
        n = 0
        if t:
            long = t["side"] == "long"
            entry = t["entry_raw"] * ((1 + slip) if long else (1 - slip))
            risk = (entry - t["stop"]) if long else (t["stop"] - entry)
            qty = math.floor(min(eq * RISK_PCT / 100 / risk, eq * MAX_POS_PCT / 100 / entry)) if risk > 0 else 0
            if qty >= 1:
                xpx = t["exit_px"]
                pnl = (xpx - entry) * qty if long else (entry - xpx) * qty
                eq += pnl
                n = 1
                trades.append({"day": d, "symbol": sym, "side": t["side"], "entry_hm": t["entry_hm"], "exit_hm": t["exit_hm"],
                               "shares": qty, "entry": round(entry, 2), "stop": round(t["stop"], 2),
                               "target": None if t["target"] is None else round(t["target"], 2), "exit": round(xpx, 2),
                               "exit_reason": t["why"], "pnl": round(pnl, 2), "r_multiple": round(pnl / (risk * qty), 2)})
        rows.append({"date": d, "equity": round(eq, 2), "trades": n})
    return trades, rows


def runner(v):
    m = v["method"]
    if m == "trend":
        return run_trend
    if m in ("orb", "gap"):
        return run_day_trades
    if m in ("sweep", "fomc_break"):
        return run_intraday_more
    return run_daily



def main():
    ap = argparse.ArgumentParser(description="Backtest gold and oil ETFs and pick each market's best method")
    ap.add_argument("--years", type=int, default=1)
    a = ap.parse_args()

    syms = list(MARKETS)
    daily = orb.daily_data(syms + ["SPY", "UUP"], a.years + 1)
    # Yahoo sometimes drops one symbol from a batch (e.g. "database is locked"): fetch it again on its own
    for s in syms + ["SPY", "UUP"]:
        for attempt in range(4):
            col = daily["Close"].get(s)
            if col is not None and col.notna().sum() > 200:
                break
            import time as _t
            _t.sleep(5 * (attempt + 1))
            one = orb.daily_data([s], a.years + 1)
            for f in daily:
                daily[f] = daily[f].drop(columns=[s], errors="ignore").join(one[f][[s]], how="outer") if s in one[f] else daily[f]
        else:
            raise SystemExit(f"No daily prices for {s} from Yahoo Finance after retries")
    closes = daily["Close"]
    today = pd.Timestamp.now(tz=orb.NY).strftime("%Y-%m-%d")
    idx = [d for d in closes.index if d.strftime("%Y-%m-%d") < today]
    window = idx[-252 * a.years:]
    days = [d.strftime("%Y-%m-%d") for d in window]

    start = (pd.Timestamp(days[0], tz=orb.NY) - pd.Timedelta(days=30)).to_pydatetime()
    end = (pd.Timestamp(days[-1], tz=orb.NY) + pd.Timedelta(hours=23)).to_pydatetime()
    bars = orb.day_bars(orb.minute_bars(syms, start, end))

    ctx = {"bars": bars, "daily": daily, "atr": {}, "prev_close": {}, "rvol": {}, "ind": {}, "prev_high": {}, "prev_low": {}}
    for s in syms:
        ctx["ind"][s] = indicators(daily, s)
        ctx["prev_high"][s] = {d.strftime("%Y-%m-%d"): float(x) for d, x in daily["High"][s].shift(1).items()}
        ctx["prev_low"][s] = {d.strftime("%Y-%m-%d"): float(x) for d, x in daily["Low"][s].shift(1).items()}
        h, l, c = daily["High"][s], daily["Low"][s], daily["Close"][s]
        tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().shift(1)          # known before the open
        ctx["atr"][s] = {d.strftime("%Y-%m-%d"): float(x) for d, x in atr.items()}
        ctx["prev_close"][s] = {d.strftime("%Y-%m-%d"): float(x) for d, x in c.shift(1).items()}
        first = pd.Series({d: b.volume[0] for (sy, d), b in bars.items() if sy == s and len(b) and b.hm[0] == "09:30"}).sort_index()
        base = first.rolling(14, min_periods=10).mean().shift(1)
        ctx["rvol"][s] = (first / base).to_dict()

    spy_shares = START_CASH / float(daily["Open"]["SPY"].loc[window[0]])
    spy = {d.strftime("%Y-%m-%d"): spy_shares * float(closes["SPY"].loc[d]) for d in window}

    grid, picked = [], {}
    for s in syms:
        res = []
        for v in versions() + more_versions():
            trades, rows = runner(v)(v, s, days, ctx)
            first_pct, second_pct, mid_day = halves(rows)
            r = {"market": s, "version": v["name"], "label": v["label"], "method": v["method"],
                 "first_half_pct": first_pct, "second_half_pct": second_pct, "split_day": mid_day,
                 "full_year_pct": round(rows[-1]["equity"] / START_CASH * 100 - 100, 2),
                 "trades": len(trades), "first_half_trades": sum(t["day"] < mid_day for t in trades),
                 "_data": (trades, rows)}
            res.append(r)
            grid.append(r)
        best = pick(res)
        picked[s] = best
        print(f"\n== {MARKETS[s]}: {len(res)} versions ==")
        for r in sorted(res, key=lambda r: -r["first_half_pct"])[:10]:
            print(f"  {r['version']:28} first half {r['first_half_pct']:+7.2f}%  second half {r['second_half_pct']:+7.2f}%  "
                  f"year {r['full_year_pct']:+7.2f}%  trades {r['trades']}")
        if best:
            print(f"  PICK (first half only): {best['version']} -> second half {best['second_half_pct']:+.2f}%")

    summ, eq_all, months_all, trades_all = [], [], [], []
    for s, best in picked.items():
        if not best:
            continue
        trades, rows = best["_data"]
        var = f"cmd_{s.lower()}"
        sr = summary_row(trades, rows, spy, f"{best['label']} on {s}; picked using only the first half of the year "
                                            f"(to {best['split_day']}): first half {best['first_half_pct']:+.2f}%, "
                                            f"second half (the honest test) {best['second_half_pct']:+.2f}%; "
                                            f"1% risk per trade with a stop, at most 20% of the account in the position, 0.05% slippage")
        sr.update(variant=var, strategy="commodities", label=f"{MARKETS[s]}: {best['label']}",
                  first_half_pct=best["first_half_pct"], second_half_pct=best["second_half_pct"])
        summ.append(sr)
        eq_all += [dict(r, variant=var, withdrawn_total=0.0, spy_equity=round(spy[r["date"]], 2)) for r in rows]
        months_all += [dict(m, variant=var) for m in months_of(rows)]
        trades_all += [dict(t, variant=var, id=f"{var}-{i}")
                       for i, t in enumerate(sorted(trades, key=lambda t: t["day"], reverse=True), 1)]

    OUT.mkdir(exist_ok=True)
    meta = [{"label": "backtest_commodities.py", "value": Path(__file__).read_text(encoding="utf-8"), "type": "code", "lang": "python"},
            {"label": "Input", "value": "Alpaca 5-minute bars and Yahoo Finance daily prices for GLD, USO, GDX, SPY and UUP; FOMC dates from federalreserve.gov"}]
    grid_rows = [{k: v for k, v in r.items() if k != "_data"} for r in grid]
    for fname, rows, m in (("cmd_summary.json", summ, meta), ("cmd_equity.json", eq_all, []),
                           ("cmd_months.json", months_all, []), ("cmd_trades.json", trades_all, []),
                           ("cmd_grid.json", grid_rows, [])):
        (OUT / fname).write_text(dt.to_json({"rows": rows, "meta": m}, separators=(",", ":")))
    print(dt.to_json([{k: r[k] for k in ("variant", "label", "total_return_incl_withdrawn_pct", "first_half_pct",
                                          "second_half_pct", "max_drawdown_pct", "trades")} for r in summ], indent=1))


if __name__ == "__main__":
    main()
