"""
Backtest of TJR's day-trading model (liquidity sweep -> market structure shift -> fair value gap),
written as exact rules from public descriptions of what he teaches:
  * snappchart.app "What Is TJR's ICT Trading Strategy?" (the 7-step sequence: bias, liquidity pool,
    sweep, structure shift, entry zone, stop beyond the sweep wick, target)
  * the open-source TradingView script "TJR Trades Strategy" (session-range sweep, pivot-based MSS,
    ATR-filtered fair value gaps)
TJR trades NQ and ES futures at the New York open. Here the ETFs that track them are used instead:
QQQ (Nasdaq-100) and SPY (S&P 500), 5-minute bars including pre-market, from Alpaca.

Rules (every one is a choice a discretionary trader would make by eye; these fix them in advance):
  1. Liquidity pools each day: the pre-market high and low (4:00-9:30 AM ET; the ETFs' stand-in for
     the Asia/London session ranges futures traders use) and the previous day's regular-session
     high and low.
  2. Killzone: setups must sweep and shift between 9:30 and 11:00 AM ET.
  3. Sweep: a 5-minute bar trades through a pool level and CLOSES back inside it. Bullish: wick below
     a low, close above it. Bearish: wick above a high, close below it. Each level can be swept once.
  4. Market structure shift (MSS): within 12 bars (1 hour) of the sweep, a bar closes beyond the most
     recent confirmed swing point (pivot strength 2) on the other side. A close back beyond the swept
     level first cancels the setup.
  5. Entry: the most recent fair value gap in the move from the sweep to the MSS (3-candle gap at
     least 0.25 x the 14-bar ATR). A limit order rests at the near edge of the gap until 11:30 AM.
     It is cancelled if price reaches the stop or the target before filling.
  6. Stop: just beyond the sweep's extreme wick (0.02%). Target: 2x the risk (2R).
  7. One trade a day: the first setup to fill, QQQ or SPY. Anything still open is closed at 3:55 PM.
  8. Size: risk RISK_PCT of the account per trade; the position's value is capped at LEVERAGE x the
     account (Alpaca's paper account allows 4x for day trades; futures traders use similar leverage).
     Fills: 0.02% slippage on market-type fills (stop, close-out); limit entries fill at their price.

Limits: ETFs instead of futures; pre-market instead of overnight futures ranges; 5-minute bars (TJR
often refines entries on 1-minute charts); no news filter; a fixed rule set cannot replicate the
judgment calls the method leaves to the trader.

Usage: python backtest_tjr.py [--years 1] [--risk 1.0] [--leverage 4]
Writes backtest_tjr_results/tjr_{summary,equity,trades,months}.json
"""

import argparse
import json
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

BASE = Path(__file__).resolve().parent
OUT = BASE / "backtest_tjr_results"
NY = "America/New_York"
SYMBOLS = ["QQQ", "SPY"]
STRICT = {"BAR_MIN": 5, "PIVOT": 2, "RECLAIM_BARS": 1, "MSS_BARS": 12, "FVG_ATR": 0.25, "ATR_N": 14,
          "STOP_BUFFER": 0.0002, "TARGET_R": 2.0, "TARGET": "R",
          "KILLZONE_END": "11:00", "ENTRY_UNTIL": "11:30", "FLATTEN": "15:55", "SLIP": 0.0002}
# Looser, closer to reading a 1-minute chart by eye: the sweep may reclaim the level within 3 bars,
# the structure shift may come within 30 minutes, smaller gaps count.
LOOSE = dict(STRICT, BAR_MIN=1, RECLAIM_BARS=3, MSS_BARS=30, FVG_ATR=0.10)
P = dict(STRICT)


# ---------------------------------------------------------------- data
def fetch_bars(years: int, bar_min: int = 5) -> pd.DataFrame:
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    load_dotenv(BASE / ".env")
    client = StockHistoricalDataClient(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY"))
    end = datetime.now(timezone.utc) - timedelta(days=1)
    start = end - timedelta(days=int(365 * years) + 20)
    df = client.get_stock_bars(StockBarsRequest(symbol_or_symbols=SYMBOLS, timeframe=TimeFrame(bar_min, TimeFrameUnit.Minute),
                                                start=start, end=end, adjustment=Adjustment.ALL,
                                                feed=DataFeed.SIP)).df.reset_index()
    ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(NY)
    df["day"], df["hm"] = ts.dt.strftime("%Y-%m-%d"), ts.dt.strftime("%H:%M")
    df = df[(df.hm >= "04:00") & (df.hm < "16:00")]
    return df[["symbol", "day", "hm", "open", "high", "low", "close", "volume"]].sort_values(["symbol", "day", "hm"])


# ---------------------------------------------------------------- setup detection (pure, unit-tested)
def pivots(hi, lo, k):
    """Confirmed swing highs/lows: index -> price, usable from index + k on."""
    ph, pl = {}, {}
    for j in range(k, len(hi) - k):
        if hi[j] == max(hi[j - k:j + k + 1]) and hi[j] > max(max(hi[j - k:j]), max(hi[j + 1:j + k + 1])):
            ph[j] = hi[j]
        if lo[j] == min(lo[j - k:j + k + 1]) and lo[j] < min(min(lo[j - k:j]), min(lo[j + 1:j + k + 1])):
            pl[j] = lo[j]
    return ph, pl


def atr(hi, lo, cl, n):
    tr = [hi[0] - lo[0]] + [max(hi[i] - lo[i], abs(hi[i] - cl[i - 1]), abs(lo[i] - cl[i - 1])) for i in range(1, len(hi))]
    return pd.Series(tr).rolling(n, min_periods=1).mean().to_numpy()


def find_setups(bars: pd.DataFrame, pools: dict) -> list[dict]:
    """bars: one symbol-day from 4:00, columns hm/open/high/low/close. pools: {'low': [levels], 'high': [levels]}.
    Returns setups in time order: side, sweep bar, MSS bar, entry, stop, target."""
    hm = bars["hm"].to_numpy()
    op, hi, lo, cl = (bars[c].to_numpy(float) for c in ("open", "high", "low", "close"))
    a = atr(hi, lo, cl, P["ATR_N"])
    ph, pl = pivots(hi, lo, P["PIVOT"])
    out, used = [], set()
    n = len(hm)
    for i in range(n):
        if hm[i] < "09:30" or hm[i] >= P["KILLZONE_END"]:
            continue
        for side, levels in (("long", pools["low"]), ("short", pools["high"])):
            for L in levels:
                if (side, L) in used:
                    continue
                pierced = lo[i] < L if side == "long" else hi[i] > L
                if not pierced:
                    continue
                used.add((side, L))                                   # each level is swept once
                # reclaim: a close back inside the level within RECLAIM_BARS bars (the piercing bar counts)
                r = None
                for j in range(i, min(i + P["RECLAIM_BARS"], n)):
                    if (side == "long" and cl[j] > L) or (side == "short" and cl[j] < L):
                        r = j
                        break
                if r is None:
                    continue                                          # it broke the level for real
                extreme = min(lo[i:r + 1]) if side == "long" else max(hi[i:r + 1])
                piv = ph if side == "long" else pl
                known = [j for j in piv if j + P["PIVOT"] <= r]
                if not known:
                    continue
                level_ms = piv[max(known)]
                for m in range(r + 1, min(r + 1 + P["MSS_BARS"], n)):
                    if hm[m] >= P["KILLZONE_END"]:
                        break
                    if (side == "long" and cl[m] < L) or (side == "short" and cl[m] > L):
                        break                                         # lost the level: no setup
                    extreme = min(extreme, lo[m]) if side == "long" else max(extreme, hi[m])
                    if (side == "long" and cl[m] > level_ms) or (side == "short" and cl[m] < level_ms):
                        gap = None
                        for k in range(m, i + 1, -1):
                            if side == "long" and lo[k] - hi[k - 2] >= P["FVG_ATR"] * a[k]:
                                gap = (k, lo[k])                      # near edge = top of the gap
                                break
                            if side == "short" and lo[k - 2] - hi[k] >= P["FVG_ATR"] * a[k]:
                                gap = (k, hi[k])                      # near edge = bottom of the gap
                                break
                        if gap:
                            entry = float(gap[1])
                            stop = extreme * (1 - P["STOP_BUFFER"]) if side == "long" else extreme * (1 + P["STOP_BUFFER"])
                            risk = entry - stop if side == "long" else stop - entry
                            if risk > 0:
                                target = entry + P["TARGET_R"] * risk if side == "long" else entry - P["TARGET_R"] * risk
                                if P["TARGET"] == "pool":
                                    # the opposite liquidity pool, if it is at least 1R away
                                    far = [x for x in (pools["high"] if side == "long" else pools["low"])
                                           if (x - entry if side == "long" else entry - x) >= risk]
                                    if far:
                                        target = min(far) if side == "long" else max(far)
                                out.append({"side": side, "level": float(L), "sweep_hm": str(hm[i]), "mss_hm": str(hm[m]),
                                            "mss_i": m, "entry": entry, "stop": float(stop), "target": float(target)})
                        break
    return out


def fill_and_exit(bars: pd.DataFrame, s: dict):
    """Limit entry from the bar after the MSS until ENTRY_UNTIL; then stop / target / 3:55 close-out.
    Returns (entry_hm, fill, exit_hm, exit_px, reason) or None if it never filled."""
    hm = bars["hm"].to_numpy()
    op, hi, lo, cl = (bars[c].to_numpy(float) for c in ("open", "high", "low", "close"))
    long = s["side"] == "long"
    k = None
    for j in range(s["mss_i"] + 1, len(hm)):
        if hm[j] >= P["ENTRY_UNTIL"]:
            return None
        if long:
            if op[j] <= s["stop"] or op[j] >= s["target"]:
                return None
            if lo[j] <= s["entry"]:
                k, fill = j, min(s["entry"], op[j])
                break
            if hi[j] >= s["target"]:
                return None                                         # ran to the target without us
        else:
            if op[j] >= s["stop"] or op[j] <= s["target"]:
                return None
            if hi[j] >= s["entry"]:
                k, fill = j, max(s["entry"], op[j])
                break
            if lo[j] <= s["target"]:
                return None
    if k is None:
        return None
    # exit: walk each bar's path (rising bar O-L-H-C, falling O-H-L-C) from the fill
    for i in range(k, len(hm)):
        if hm[i] >= P["FLATTEN"]:
            px = op[i] * ((1 - P["SLIP"]) if long else (1 + P["SLIP"]))
            return str(hm[k]), fill, str(hm[i]), px, "close"
        path = [op[i], lo[i], hi[i], cl[i]] if cl[i] >= op[i] else [op[i], hi[i], lo[i], cl[i]]
        if i == k:
            path = [fill] + path[1:]
        elif (long and op[i] <= s["stop"]) or (not long and op[i] >= s["stop"]):
            return str(hm[k]), fill, str(hm[i]), op[i] * ((1 - P["SLIP"]) if long else (1 + P["SLIP"])), "stop"
        elif (long and op[i] >= s["target"]) or (not long and op[i] <= s["target"]):
            return str(hm[k]), fill, str(hm[i]), op[i], "target"
        prev = path[0]
        for p in path[1:]:
            lo_, hi_ = min(prev, p), max(prev, p)
            hits = []
            if lo_ <= s["stop"] <= hi_:
                hits.append((abs(s["stop"] - prev), "stop"))
            if lo_ <= s["target"] <= hi_:
                hits.append((abs(s["target"] - prev), "target"))
            if hits:
                why = min(hits)[1]
                if why == "stop":
                    return str(hm[k]), fill, str(hm[i]), s["stop"] * ((1 - P["SLIP"]) if long else (1 + P["SLIP"])), "stop"
                return str(hm[k]), fill, str(hm[i]), s["target"], "target"
            prev = p
    return str(hm[k]), fill, str(hm[-1]), cl[-1], "close"


# ---------------------------------------------------------------- simulation
def run(years: int, risk_pct: float, leverage: float, start_cash: float = 100_000.0, bars=None):
    import yfinance as yf

    bars = bars if bars is not None else fetch_bars(years, P["BAR_MIN"])
    days_all = sorted(bars.day.unique())
    today = pd.Timestamp.now(tz=NY).strftime("%Y-%m-%d")
    days = [d for d in days_all if d < today][-252 * years:]
    spy_d = yf.Ticker("SPY").history(start=days[0], end=(pd.Timestamp(days[-1]) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                                     interval="1d", auto_adjust=True)
    spy_d.index = spy_d.index.strftime("%Y-%m-%d")
    by = {k: g.reset_index(drop=True) for k, g in bars.groupby(["symbol", "day"])}

    cash, trades, eq_rows, months, month_pnl = start_cash, [], [], [], 0.0
    setups_seen = no_fill = 0
    for di, d in enumerate(days):
        prev = days_all[days_all.index(d) - 1] if days_all.index(d) > 0 else None
        candidates = []
        for sym in SYMBOLS:
            b = by.get((sym, d))
            pb = by.get((sym, prev)) if prev else None
            if b is None or pb is None:
                continue
            pm = b[b.hm < "09:30"]
            prth = pb[(pb.hm >= "09:30") & (pb.hm < "16:00")]
            if pm.empty or prth.empty:
                continue
            pools = {"low": sorted({float(pm.low.min()), float(prth.low.min())}),
                     "high": sorted({float(pm.high.max()), float(prth.high.max())})}
            for s in find_setups(b, pools):
                setups_seen += 1
                r = fill_and_exit(b, s)
                if r is None:
                    no_fill += 1
                    continue
                candidates.append((r[0], sym, s, r))
        day_pnl, taken = 0.0, None
        if candidates:
            candidates.sort(key=lambda c: (c[0], c[1]))            # the first setup to fill that day
            entry_hm, sym, s, (ehm, fill, xhm, xpx, why) = candidates[0]
            long = s["side"] == "long"
            per_share = abs(fill - s["stop"])
            qty = math.floor(min(cash * risk_pct / 100 / per_share, cash * leverage / fill))
            if qty >= 1:
                pnl = (xpx - fill) * qty if long else (fill - xpx) * qty
                day_pnl = pnl
                taken = {"day": d, "symbol": sym, "side": s["side"], "level_swept": round(s["level"], 2),
                         "sweep_hm": s["sweep_hm"], "mss_hm": s["mss_hm"], "entry_hm": ehm, "exit_hm": xhm,
                         "shares": qty, "entry": round(fill, 2), "stop": round(s["stop"], 2), "target": round(s["target"], 2),
                         "exit": round(float(xpx), 2), "exit_reason": why, "pnl": round(float(pnl), 2),
                         "r_multiple": round(float(pnl / (per_share * qty)), 2), "position_usd": round(qty * fill, 2)}
                trades.append(taken)
        cash += day_pnl
        month_pnl += day_pnl
        if di == len(days) - 1 or days[di + 1][:7] != d[:7]:
            months.append({"month": d[:7], "realized_pl": round(month_pnl, 2), "account_value": round(cash, 2)})
            month_pnl = 0.0
        spy_eq = start_cash / float(spy_d["Open"].iloc[0]) * float(spy_d["Close"].get(d, np.nan)) if d in spy_d.index else None
        eq_rows.append({"date": d, "equity": round(cash, 2), "trades": 1 if taken else 0,
                        "spy_equity": round(spy_eq, 2) if spy_eq else None})

    eq = pd.Series([r["equity"] for r in eq_rows])
    spy = pd.Series([r["spy_equity"] for r in eq_rows]).ffill()
    dd = lambda x: round(float((x / x.cummax() - 1).min() * 100), 2)  # noqa: E731
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    gw, gl = sum(t["pnl"] for t in wins), -sum(t["pnl"] for t in losses)
    summary = {
        "start": days[0], "end": days[-1], "trading_days": len(days), "start_value": start_cash,
        "end_value": round(cash, 2), "total_return_pct": round((cash / start_cash - 1) * 100, 2),
        "max_drawdown_pct": dd(eq), "spy_end_value": round(float(spy.iloc[-1]), 2),
        "spy_total_return_pct": round((float(spy.iloc[-1]) / start_cash - 1) * 100, 2), "spy_max_drawdown_pct": dd(spy),
        "trades": len(trades), "days_with_trade_pct": round(len(trades) / len(days) * 100, 1),
        "setups_found": setups_seen, "setups_never_filled": no_fill,
        "wins": len(wins), "losses": len(losses),
        "win_rate_pct": round(len(wins) / len(trades) * 100, 1) if trades else None,
        "avg_win": round(gw / len(wins), 2) if wins else None,
        "avg_loss": round(-gl / len(losses), 2) if losses else None,
        "profit_factor": round(gw / gl, 2) if gl else None,
        "avg_r": round(sum(t["r_multiple"] for t in trades) / len(trades), 3) if trades else None,
        "exits": {r: sum(t["exit_reason"] == r for t in trades) for r in ("stop", "target", "close")},
        "longs": sum(t["side"] == "long" for t in trades), "shorts": sum(t["side"] == "short" for t in trades),
        "by_symbol": {s: sum(t["symbol"] == s for t in trades) for s in SYMBOLS},
        "avg_position_usd": round(sum(t["position_usd"] for t in trades) / len(trades), 2) if trades else None,
        "months_positive": sum(m["realized_pl"] > 0 for m in months), "months": len(months),
        "best_month": max(months, key=lambda m: m["realized_pl"]) if months else None,
        "worst_month": min(months, key=lambda m: m["realized_pl"]) if months else None,
        "risk_pct": risk_pct, "leverage": leverage, "rules": dict(P),
        "run_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    return summary, eq_rows, trades, months


VARIANTS = {
    "strict_5m": ("Strict rules, 5-minute bars, 2R target", STRICT),
    "loose_1m": ("Looser rules, 1-minute bars, 2R target", LOOSE),
    "loose_1m_pool": ("Looser rules, 1-minute bars, target the opposite liquidity level", dict(LOOSE, TARGET="pool")),
}


def main():
    ap = argparse.ArgumentParser(description="Backtest of TJR's sweep / structure shift / FVG model on QQQ and SPY")
    ap.add_argument("--years", type=int, default=1)
    ap.add_argument("--risk", type=float, default=1.0, help="%% of the account risked per trade")
    ap.add_argument("--leverage", type=float, default=4.0, help="largest position as a multiple of the account")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    meta = [{"label": "backtest_tjr.py", "value": Path(__file__).read_text(encoding="utf-8"), "type": "code", "lang": "python"}]
    cache = {}
    for vname, (label, params) in VARIANTS.items():
        P.clear()
        P.update(params)
        if params["BAR_MIN"] not in cache:
            cache[params["BAR_MIN"]] = fetch_bars(a.years, params["BAR_MIN"])
        for lname, lev in (("lev4", a.leverage), ("lev1", 1.0)):
            s, eq, tr, mo = run(a.years, a.risk, lev, bars=cache[params["BAR_MIN"]])
            s["variant"], s["label"] = f"{vname}_{lname}", f"{label}, positions up to {lev:g}x the account"
            for fname, rows, m in (("summary", [s], meta), ("equity", eq, []), ("trades", tr, []), ("months", mo, [])):
                (OUT / f"tjr_{vname}_{lname}_{fname}.json").write_text(
                    json.dumps({"rows": rows, "meta": m}, default=str, separators=(",", ":")))
            print(f"\n== {s['label']} ==")
            print(json.dumps({k: s[k] for k in ("end_value", "total_return_pct", "max_drawdown_pct", "spy_total_return_pct",
                                                "trades", "setups_found", "setups_never_filled", "win_rate_pct",
                                                "avg_win", "avg_loss", "profit_factor", "avg_r", "exits",
                                                "months_positive", "months")}, default=str))


if __name__ == "__main__":
    main()
