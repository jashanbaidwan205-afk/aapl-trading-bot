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
MARKETS = {"GLD": "Gold (GLD)", "USO": "Oil (USO)"}
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
        o, c = b.open[0], b.close[0]
        if c == o:
            return None
        side = "long" if c > o else "short"
        level = b.high[0] if side == "long" else b.low[0]
        hit = (b.high[1:] >= level) if side == "long" else (b.low[1:] <= level)
        hit &= b.hm[1:] < ENTRY_UNTIL
        idx = np.flatnonzero(hit)
        if not len(idx):
            return None
        k = int(idx[0]) + 1
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
    method_min_trades = method_min_trades or {"orb": 15, "gap": 8, "trend": 1}
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


def main():
    ap = argparse.ArgumentParser(description="Backtest gold and oil ETFs and pick each market's best method")
    ap.add_argument("--years", type=int, default=1)
    a = ap.parse_args()

    syms = list(MARKETS)
    daily = orb.daily_data(syms + ["SPY"], a.years + 1)
    closes = daily["Close"]
    today = pd.Timestamp.now(tz=orb.NY).strftime("%Y-%m-%d")
    idx = [d for d in closes.index if d.strftime("%Y-%m-%d") < today]
    window = idx[-252 * a.years:]
    days = [d.strftime("%Y-%m-%d") for d in window]

    start = (pd.Timestamp(days[0], tz=orb.NY) - pd.Timedelta(days=30)).to_pydatetime()
    end = (pd.Timestamp(days[-1], tz=orb.NY) + pd.Timedelta(hours=23)).to_pydatetime()
    bars = orb.day_bars(orb.minute_bars(syms, start, end))

    ctx = {"bars": bars, "daily": daily, "atr": {}, "prev_close": {}, "rvol": {}}
    for s in syms:
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
        for v in versions():
            trades, rows = (run_trend if v["method"] == "trend" else run_day_trades)(v, s, days, ctx)
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
        for r in sorted(res, key=lambda r: -r["first_half_pct"])[:6]:
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
                                            f"1% risk per day trade, at most 20% of the account in the position, 0.05% slippage")
        sr.update(variant=var, strategy="commodities", label=f"{MARKETS[s]}: {best['label']}",
                  first_half_pct=best["first_half_pct"], second_half_pct=best["second_half_pct"])
        summ.append(sr)
        eq_all += [dict(r, variant=var, withdrawn_total=0.0, spy_equity=round(spy[r["date"]], 2)) for r in rows]
        months_all += [dict(m, variant=var) for m in months_of(rows)]
        trades_all += [dict(t, variant=var, id=f"{var}-{i}")
                       for i, t in enumerate(sorted(trades, key=lambda t: t["day"], reverse=True), 1)]

    OUT.mkdir(exist_ok=True)
    meta = [{"label": "backtest_commodities.py", "value": Path(__file__).read_text(encoding="utf-8"), "type": "code", "lang": "python"},
            {"label": "Input", "value": "Alpaca 5-minute bars and Yahoo Finance daily prices for GLD, USO and SPY"}]
    grid_rows = [{k: v for k, v in r.items() if k != "_data"} for r in grid]
    for fname, rows, m in (("cmd_summary.json", summ, meta), ("cmd_equity.json", eq_all, []),
                           ("cmd_months.json", months_all, []), ("cmd_trades.json", trades_all, []),
                           ("cmd_grid.json", grid_rows, [])):
        (OUT / fname).write_text(dt.to_json({"rows": rows, "meta": m}, separators=(",", ":")))
    print(dt.to_json([{k: r[k] for k in ("variant", "label", "total_return_incl_withdrawn_pct", "first_half_pct",
                                          "second_half_pct", "max_drawdown_pct", "trades")} for r in summ], indent=1))


if __name__ == "__main__":
    main()
